from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .data import (
    ArrayDataset,
    RANDataset,
    gaussian_config_from_run_config,
    load_jet_dataset,
)
from .rantypes import RUN_DIR, DatasetName

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from logging import Logger
    from pathlib import Path

    from jax import Array as JaxArray
    from numpy.typing import NDArray

    from .rantypes import ZXY, DatasetSplits, GaussianConfig, Populations, RANModel


logger: Logger = logging.getLogger(name=__name__)


def apply_to_runs(
    run_dir: Path,
    evaluate_one: Callable[[Path], object],
    description: str,
    log: logging.Logger,
) -> None:
    if (run_dir / "config.json").exists():
        evaluate_one(run_dir)
        return

    run_dirs: list[Path] = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
    )
    log.info("Found %d runs to %s", len(run_dirs), description)
    for d in run_dirs:
        try:
            evaluate_one(d)
        except Exception:
            log.warning("%s: failed", d.name, exc_info=True)


def _load_splits(config: dict, dtype=np.double) -> DatasetSplits[np.double]:
    dataset: DatasetName = DatasetName(
        value=config.get("dataset", DatasetName.gaussian.value)
    )
    n_samples: int = config["n_samples"]
    batch_size: int = config["batch_size"]
    dim: int = config["dim"]
    data_seed: int = config.get("data_seed", 42)

    logger.info("Loading dataset: %s", dataset)
    if dataset == DatasetName.gaussian:
        if "gaussian_params" in config:
            params: GaussianConfig = gaussian_config_from_run_config(
                config["gaussian_params"], dim
            )
        else:
            # Legacy config format: hardcoded mu/sigma, only smearing varied.
            # Raw sigmas, so they go through the same promotion.
            params: GaussianConfig = gaussian_config_from_run_config(
                {
                    "mu_gen": [0.5] * dim,
                    "mu_true": [0.0] * dim,
                    "sigma_gen": 0.9,
                    "sigma_true": 1.0,
                    "sigma_detector": config.get("smearing", 0.5),
                },
                dim,
            )
        return RANDataset(batch_size, data_seed, dtype=dtype).generate_gaussian_dataset(
            params=params,
            n_samples=n_samples,
        )
    if dataset == DatasetName.jets:
        splits, _, _ = load_jet_dataset(
            n_samples,
            batch_size,
            variables=frozenset(config["variables"]),
            seed=data_seed,
        )
        return splits
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _collect_test_data[T: np.floating = np.double](test_ds: ArrayDataset[T]) -> ZXY[T]:
    """Return the test split as one flat labelled sample."""
    return test_ds.as_arrays()


def _get_weights(
    g: RANModel, z_gen: NDArray, chunk_size: int = 10_000
) -> NDArray[np.double]:
    """Compute normalized generator weights, mean 1, as host NumPy.

    Chunked because it is the intermediate activations, not the output, that set
    peak memory --- a full-dataset forward pass through a wide hidden layer is
    orders of magnitude larger than the one weight per event it produces. The
    chunks are stitched together and normalized on device, so the whole call
    costs exactly one device-to-host copy rather than one per chunk.
    """
    n: int = len(z_gen)
    chunks: list[JaxArray] = [
        jnp.ravel(g(z_gen[start : start + chunk_size]))
        for start in range(0, n, chunk_size)
    ]
    raw: JaxArray = jnp.concatenate(chunks)
    return np.asarray(a=raw / (jnp.sum(raw) / n), dtype=np.double)


def _dim(x: NDArray, /) -> int:
    return x.shape[1] if x.ndim > 1 else 1


def _wd_per_dim[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
) -> NDArray[np.double]:
    """1D Wasserstein distance per dimension using sorted-CDF fast path."""
    dim: int = _dim(ref)
    result: NDArray[np.double] = np.empty(shape=dim, dtype=np.double)
    if dim > 1:
        for i in range(dim):
            r: NDArray[T] = ref[:, i]
            c: NDArray[T] = comp[:, i]
            result[i] = wasserstein_distance(r, c, v_weights=weights)
    else:
        result[0] = wasserstein_distance(ref.ravel(), comp.ravel(), v_weights=weights)
    return result


def _normalized_histograms[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> Iterator[tuple[NDArray[np.double], NDArray[np.double]]]:
    # A flat 1D sample is one feature, not `dim` scalar features, so it is
    # treated as a single column -- but only when both sides agree on that;
    # a 1D/2D mismatch stays an error rather than silently reshaping one side.
    ref_2d: NDArray[T] = ref.reshape(-1, 1) if ref.ndim == 1 and comp.ndim == 1 else ref
    comp_2d: NDArray[T] = (
        comp.reshape(-1, 1) if ref.ndim == 1 and comp.ndim == 1 else comp
    )
    dim: int = _dim(ref_2d)
    for i in range(dim):
        r: NDArray[T] = ref_2d[:, i]
        c: NDArray[T] = comp_2d[:, i]

        bins: NDArray[np.double] = np.linspace(
            start=min(r.min(), c.min()), stop=max(r.max(), c.max()), num=n_bins + 1
        )
        # weights=None is np.histogram's own default, so the unweighted and
        # weighted cases need no branch here.
        h_ref: NDArray[np.intp] = np.histogram(a=r, bins=bins)[0]
        h_comp: NDArray[np.intp | T] = np.histogram(a=c, bins=bins, weights=weights)[0]
        yield h_ref / (h_ref.sum() or 1.0), np.divide(h_comp, h_comp.sum() or 1.0)


def _js_per_dim[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    return np.array(
        [
            jensenshannon(p, q) ** 2
            for p, q in _normalized_histograms(ref, comp, weights, n_bins)
        ]
    )


def _triangular_per_dim[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    """Triangular discriminator (Vincze-LeCam divergence) per dimension.

    Δ(p,q) = Σ (p_i - q_i)² / (p_i + q_i)  ×  1e3

    where p_i, q_i are histogram probability masses. The bin-width factor
    cancels analytically, so this works directly on normalized histograms.
    """
    dim: int = _dim(ref)
    result: NDArray[np.double] = np.empty(shape=dim, dtype=np.double)
    for i, (p, q) in enumerate(_normalized_histograms(ref, comp, weights, n_bins)):
        denom: NDArray[np.double] = p + q
        mask: NDArray[np.bool] = denom > 0
        diff: NDArray[np.double] = p - q
        result[i] = np.sum(a=diff[mask] ** 2 / denom[mask]) * 1e3
    return result


def _improvement(before: float, after: float) -> float:
    return (1 - after / before) * 100 if before > 0 else 0.0


def evaluate_run(run_dir: Path, force: bool = False) -> dict:
    """Evaluate a single run directory."""
    out_path: Path = run_dir / "metrics.json"

    if out_path.exists() and not force:
        logger.info("%s: metrics.json exists, skipping (use --force)", run_dir.name)
        return json.loads(out_path.read_text())

    # Imported here, not at module scope, so this module stays keras-free on
    # import. ran.baselines.omnifold depends on that: it must pin
    # KERAS_BACKEND=tensorflow before anything pulls keras in, and it imports
    # from this module.
    import keras

    config = json.loads((run_dir / "config.json").read_text())
    logger.info("%s: loading model and data...", run_dir.name)
    g: RANModel = keras.saving.load_model(run_dir / "generator.keras")

    splits: DatasetSplits = _load_splits(config)
    test: Populations = _collect_test_data(splits.test).partition()
    w = _get_weights(g, test.mc.z)

    # Variable names for labeling
    dataset = config.get("dataset", "gaussian")
    dim = config["dim"]
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(dim)]

    metrics: dict = {}

    for level, data, mc in [
        ("detector", test.data, test.mc.x),
        ("particle", test.require_truth(), test.mc.z),
    ]:
        wd_before: NDArray[np.double] = _wd_per_dim(ref=data, comp=mc)
        wd_after: NDArray[np.double] = _wd_per_dim(ref=data, comp=mc, weights=w)
        js_before: NDArray[np.double] = _js_per_dim(ref=data, comp=mc)
        js_after: NDArray[np.double] = _js_per_dim(ref=data, comp=mc, weights=w)
        td_before: NDArray[np.double] = _triangular_per_dim(ref=data, comp=mc)
        td_after: NDArray[np.double] = _triangular_per_dim(ref=data, comp=mc, weights=w)

        for i, var in enumerate(var_names):
            key: str = f"{level}_{var}"
            metrics[key] = {
                "wasserstein_before": wd_before[i],
                "wasserstein_after": wd_after[i],
                "wasserstein_improvement_pct": _improvement(wd_before[i], wd_after[i]),
                "jensenshannon_before": js_before[i],
                "jensenshannon_after": js_after[i],
                "jensenshannon_improvement_pct": _improvement(
                    js_before[i], js_after[i]
                ),
                "triangular_before": td_before[i],
                "triangular_after": td_after[i],
                "triangular_improvement_pct": _improvement(td_before[i], td_after[i]),
            }

    json.dump(obj=metrics, fp=out_path.open("w"), indent=2)
    logger.info("%s: saved metrics to %s", run_dir.name, out_path)
    render_metrics(run_dir.name, metrics, var_names)
    return metrics


def render_metrics(
    run_name: str,
    metrics: dict,
    var_names: list[str],
    console: Console | None = None,
    /,
) -> None:
    """Render evaluation metrics as one Rich table per available level."""
    active_console: Console = console or Console()
    for level in ("detector", "particle"):
        level_metrics = [
            (var, metrics[f"{level}_{var}"])
            for var in var_names
            if f"{level}_{var}" in metrics
        ]
        if not level_metrics:
            continue
        table = Table(title=f"{run_name} — {level.title()} level")
        table.add_column(header="Variable")
        table.add_column(header="Metric")
        table.add_column(header="Before", justify="right")
        table.add_column(header="After", justify="right")
        table.add_column(header="Improvement", justify="right")
        for var, m in level_metrics:
            table.add_row(
                var,
                "Wasserstein",
                f"{m['wasserstein_before']:.4f}",
                f"{m['wasserstein_after']:.4f}",
                f"{m['wasserstein_improvement_pct']:+.1f}%",
            )
            table.add_row(
                "",
                "JS div",
                f"{m['jensenshannon_before']:.6f}",
                f"{m['jensenshannon_after']:.6f}",
                f"{m['jensenshannon_improvement_pct']:+.1f}%",
            )
            table.add_row(
                "",
                "Delta (x1e3)",
                f"{m['triangular_before']:.4f}",
                f"{m['triangular_after']:.4f}",
                f"{m['triangular_improvement_pct']:+.1f}%",
            )
        active_console.print(table)


def evaluate_runs(run_dir: Path = RUN_DIR, force: bool = False) -> None:
    """Compute distance metrics for completed runs.

    Args:
        run_dir: Path to a single run or a directory containing multiple runs.
        force: Recompute even if metrics.json already exists.
    """
    apply_to_runs(
        run_dir,
        evaluate_one=lambda d: evaluate_run(run_dir=d, force=force),
        description="evaluate",
        log=logger,
    )
