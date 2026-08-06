"""Compute distance metrics on test sets for completed runs.

Computes per-dimension 1D Wasserstein distances and Jensen-Shannon
divergences, both before and after reweighting. Uses only memory-efficient
algorithms: sorted-CDF Wasserstein (O(n log n)) and histogram-based JS
divergence.

Usage:
    uv run -m ran evaluate                          # all runs in runs/
    uv run -m ran evaluate --run-dir runs/2026-...  # single run
    uv run -m ran evaluate --force                  # recompute existing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy import ndarray
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from logging import Logger

    import keras

    from .rantypes import DatasetSplits


logger: Logger = logging.getLogger(__name__)


def apply_to_runs(
    run_dir: str | Path,
    evaluate_one: Callable[[Path], object],
    description: str,
    log: logging.Logger,
) -> None:
    """Apply `evaluate_one` to a single run directory, or to every run inside one.

    A directory is a run if it holds a config.json; otherwise it is treated as a
    parent of runs. In the multi-run case one failure is logged and skipped
    rather than abandoning the remaining runs -- the baselines are long jobs and
    a partial sweep is more useful than none.
    """
    run_dir = Path(run_dir)
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


def _load_splits(config: dict) -> DatasetSplits:
    """Reconstruct dataset splits from a run config.

    Must reproduce the split the run trained on, so the dataset seed comes from
    the config. Runs predating seed recording used the then-hardcoded 42.
    """
    dataset = config.get("dataset", "gaussian")
    n_samples = config["n_samples"]
    batch_size = config["batch_size"]
    dim = config["dim"]
    data_seed = config.get("data_seed", 42)

    logger.info("Loading dataset: %s", dataset)
    if dataset == "gaussian":
        if "gaussian_params" in config:
            params = gaussian_config_from_run_config(config["gaussian_params"], dim)
        else:
            # Legacy config format: hardcoded mu/sigma, only smearing varied.
            # Raw sigmas, so they go through the same promotion.
            params = gaussian_config_from_run_config(
                {
                    "mu_gen": [0.5] * dim,
                    "mu_true": [0.0] * dim,
                    "sigma_gen": 0.9,
                    "sigma_true": 1.0,
                    "sigma_detector": config.get("smearing", 0.5),
                },
                dim,
            )
        return RANDataset(
            batch_size=batch_size, seed=data_seed
        ).generate_gaussian_dataset(
            params=params,
            n_samples=n_samples,
        )
    if dataset == "jets":
        splits, _, _ = load_jet_dataset(
            n_samples=n_samples,
            batch_size=batch_size,
            variables=frozenset(config["variables"]),
            seed=data_seed,
        )
        return splits
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _collect_test_data(test_ds: ArrayDataset) -> tuple[ndarray, ndarray, ndarray]:
    """Return the test split as flat (z, x, y) arrays."""
    return test_ds.as_arrays()


def _get_weights(
    g: keras.Model, z_gen: npt.NDArray, chunk_size: int = 10_000
) -> npt.NDArray:
    """Compute normalized generator weights in chunks to limit peak memory."""
    n = len(z_gen)
    raw = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        raw[start:end] = np.asarray(g(z_gen[start:end])).flatten()
    return raw / raw.mean()


def _wd_per_dim(
    ref: npt.NDArray,
    comp: npt.NDArray,
    weights: npt.NDArray | None = None,
) -> list[float]:
    """1D Wasserstein distance per dimension using sorted-CDF fast path."""
    dim = ref.shape[1] if ref.ndim > 1 else 1
    result: list[float] = []
    for i in range(dim):
        r = ref[:, i] if dim > 1 else ref.ravel()
        c = comp[:, i] if dim > 1 else comp.ravel()
        result.append(float(wasserstein_distance(r, c, v_weights=weights)))
    return result


def _normalized_histograms(
    ref: npt.NDArray,
    comp: npt.NDArray,
    weights: npt.NDArray | None = None,
    n_bins: int = 100,
) -> Iterator[tuple[npt.NDArray[np.double], npt.NDArray[np.double]]]:
    """Yield the (p, q) probability histograms for each dimension of ref/comp.

    Both histograms share one binning per dimension -- `n_bins` uniform bins
    over the combined range -- which is what makes the divergences below
    comparable across dimensions. `weights` reweights `comp` only, and an
    all-zero histogram is left unnormalized rather than divided by zero.
    """
    dim = ref.shape[1] if ref.ndim > 1 else 1
    for i in range(dim):
        r = ref[:, i] if dim > 1 else ref.ravel()
        c = comp[:, i] if dim > 1 else comp.ravel()

        bins = np.linspace(min(r.min(), c.min()), max(r.max(), c.max()), n_bins + 1)
        # weights=None is np.histogram's own default, so the unweighted and
        # weighted cases need no branch here.
        h_ref = np.histogram(r, bins=bins)[0].astype(np.double)
        h_comp = np.histogram(c, bins=bins, weights=weights)[0].astype(np.double)

        s_ref = h_ref.sum()
        s_comp = h_comp.sum()
        if s_ref > 0:
            h_ref /= s_ref
        if s_comp > 0:
            h_comp /= s_comp
        yield h_ref, h_comp


def _js_per_dim(
    ref: npt.NDArray,
    comp: npt.NDArray,
    weights: npt.NDArray | None = None,
    n_bins: int = 100,
) -> list[float]:
    """Jensen-Shannon divergence per dimension via histogramming.

    Returns JS divergence (squared JS distance) per dimension.
    """
    return [
        float(jensenshannon(p, q) ** 2)
        for p, q in _normalized_histograms(ref, comp, weights, n_bins)
    ]


def _triangular_per_dim(
    ref: npt.NDArray,
    comp: npt.NDArray,
    weights: npt.NDArray | None = None,
    n_bins: int = 100,
) -> list[float]:
    """Triangular discriminator (Vincze-LeCam divergence) per dimension.

    Δ(p,q) = Σ (p_i - q_i)² / (p_i + q_i)  ×  1e3

    where p_i, q_i are histogram probability masses. The bin-width factor
    cancels analytically, so this works directly on normalized histograms.
    """
    result: list[float] = []
    for p, q in _normalized_histograms(ref, comp, weights, n_bins):
        denom = p + q
        mask = denom > 0
        diff = p - q
        result.append(float(np.sum(diff[mask] ** 2 / denom[mask]) * 1e3))
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
    g = keras.saving.load_model(run_dir / "generator.keras")

    splits: DatasetSplits = _load_splits(config)
    z, x, y = _collect_test_data(splits.test)

    z_data, z_mc = z[y == 1], z[y == 0]
    x_data, x_mc = x[y == 1], x[y == 0]
    w = _get_weights(g, z_mc)

    # Variable names for labeling
    dataset = config.get("dataset", "gaussian")
    dim = config["dim"]
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(dim)]

    metrics: dict = {}

    for level, data, mc in [("detector", x_data, x_mc), ("particle", z_data, z_mc)]:
        wd_before: list[float] = _wd_per_dim(data, mc)
        wd_after: list[float] = _wd_per_dim(data, mc, weights=w)
        js_before: list[float] = _js_per_dim(data, mc)
        js_after: list[float] = _js_per_dim(data, mc, weights=w)
        td_before: list[float] = _triangular_per_dim(data, mc)
        td_after: list[float] = _triangular_per_dim(data, mc, weights=w)

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

    json.dump(metrics, out_path.open("w"), indent=2)
    logger.info("%s: saved metrics to %s", run_dir.name, out_path)
    render_metrics(run_dir.name, metrics, var_names)
    return metrics


def render_metrics(
    run_name: str,
    metrics: dict,
    var_names: list[str],
    console: Console | None = None,
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
        table.add_column("Variable")
        table.add_column("Metric")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Improvement", justify="right")
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


def evaluate_runs(run_dir: str | Path = "runs", force: bool = False) -> None:
    """Compute distance metrics for completed runs.

    Args:
        run_dir: Path to a single run or a directory containing multiple runs.
        force: Recompute even if metrics.json already exists.
    """
    apply_to_runs(run_dir, lambda d: evaluate_run(d, force=force), "evaluate", logger)
