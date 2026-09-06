from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from .data import (
    ArrayDataset,
    RANDataset,
    gaussian_config_from_run_config,
    load_jet_dataset,
)
from .rantypes import EVENT_DTYPE, RUN_DIR, DatasetName

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from jax import Array as JaxArray
    from numpy.typing import NDArray

    from .rantypes import (
        ZXY,
        DatasetSplits,
        EventArray,
        GaussianConfig,
        Populations,
        RANModel,
    )


logger: Logger = logging.getLogger(name=__name__)


def apply_to_runs(
    run_dir: Path,
    evaluate_one: Callable[[Path], object],
    description: str,
    log: logging.Logger,
) -> None:
    if (run_dir / "config.json").exists():
        _ = evaluate_one(run_dir)
        return

    run_dirs: list[Path] = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
    )
    log.info("Found %d runs to %s", len(run_dirs), description)
    for d in run_dirs:
        try:
            _ = evaluate_one(d)
        except Exception:
            log.warning("%s: failed", d.name, exc_info=True)


def _load_splits(config: dict[str, Any]) -> DatasetSplits:
    dataset: DatasetName = DatasetName(
        value=str(config.get("dataset", DatasetName.gaussian.value))
    )
    n_samples: int = config["n_samples"]
    batch_size: int = config["batch_size"]
    dim: int = config["dim"]
    data_seed: int = config.get("data_seed", 42)

    logger.info("Loading dataset: %s", dataset)
    if dataset == DatasetName.gaussian:
        if "gaussian_params" in config:
            params: GaussianConfig = gaussian_config_from_run_config(
                cast("Mapping[str, Any]", config["gaussian_params"]), dim
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
        return RANDataset(batch_size, data_seed).generate_gaussian_dataset(
            params=params,
            n_samples=n_samples,
        )
    if dataset == DatasetName.jets:
        splits, _, _ = load_jet_dataset(
            n_samples,
            batch_size,
            # The recorded list, in the recorded order. Round-tripping it
            # through a set here is what mismatched these columns against the
            # `var_names` below --- and against the generator's own training.
            variables=cast("Sequence[str]", config["variables"]),
            seed=data_seed,
        )
        return splits
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _collect_test_data(test_ds: ArrayDataset) -> ZXY:
    """Return the test split as one flat labelled sample."""
    return test_ds.as_arrays()


def _generator_weights(
    g: RANModel, z_gen: NDArray[Any], chunk_size: int = 10_000
) -> JaxArray:
    """Normalized generator weights, mean 1, left on device.

    Chunked because it is the intermediate activations, not the output, that set
    peak memory --- a full-dataset forward pass through a wide hidden layer is
    orders of magnitude larger than the one weight per event it produces. The
    chunks are stitched together and normalized on device, so nothing crosses
    the boundary here at all.
    """
    n: int = len(z_gen)
    chunks: list[JaxArray] = [
        jnp.ravel(g(z_gen[start : start + chunk_size]))
        for start in range(0, n, chunk_size)
    ]
    raw: JaxArray = jnp.concatenate(chunks)
    return raw / (jnp.sum(raw) / n)


def _get_weights(
    g: RANModel, z_gen: NDArray[Any], chunk_size: int = 10_000
) -> EventArray:
    """`_generator_weights`, copied back to the host.

    The callers that want NumPy --- plotting, and the uncertainty design, which
    stacks weight vectors across cells --- go through here and pay exactly one
    device-to-host copy. `evaluate_run` does not: every metric it feeds runs on
    device, so it keeps the device array.
    """
    return np.asarray(a=_generator_weights(g, z_gen, chunk_size))


def _dim(x: NDArray[Any], /) -> int:
    return x.shape[1] if x.ndim > 1 else 1


def _as_columns(ref: EventArray, comp: EventArray) -> tuple[EventArray, EventArray]:
    """Both samples as `(n, dim)`, so every kernel below is one shape.

    A flat 1D sample is one feature, not `dim` scalar features, so it becomes a
    single column -- but only when both sides agree on that. A 1D/2D mismatch is
    a caller error rather than something to silently reshape one side of.
    """
    if ref.ndim != comp.ndim:
        raise ValueError(
            f"ref and comp must have the same rank; got {ref.ndim} and {comp.ndim}"
        )
    if ref.ndim == 1:
        return ref.reshape(-1, 1), comp.reshape(-1, 1)
    return ref, comp


def _prepare(
    ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Both samples as device `(n, dim)` arrays, plus the weight vector.

    Unweighted is `weights = 1`, not a separate branch: a `None` would be a
    different trace and so a second XLA compile of every kernel below, for a
    multiplication by one.
    """
    ref_2d, comp_2d = _as_columns(ref, comp)
    w: JaxArray = (
        jnp.ones((comp_2d.shape[0],), dtype=EVENT_DTYPE)
        if weights is None
        else jnp.asarray(weights)
    )
    return jnp.asarray(ref_2d), jnp.asarray(comp_2d), w


def _bin_edges(
    ref: EventArray | JaxArray, comp: EventArray | JaxArray, n_bins: int
) -> NDArray[np.single]:
    """`(dim, n_bins + 1)` uniform edges over each dimension's combined range.

    Built on the host with `np.linspace` rather than on device, because the
    edges are the one thing both histograms have to agree on bit-for-bit and
    recomputing them as `lo + (hi - lo) * t` rounds differently in the last ulp
    -- enough to drop a value sitting on a boundary into the neighbouring bin.
    Only the `2 x dim` extrema cross back to get them, so this costs nothing.

    They come back **float32**, and that is not incidental: `JAX_ENABLE_X64=0`
    truncates a float64 array on its way into a traced function, so float64
    edges would be re-rounded at that boundary and the bin a value lands in
    would stop matching the edges the host computed. Deciding the width in the
    dtype the comparison happens in is what keeps the two ends one function.
    """
    lo: NDArray[np.single] = np.asarray(
        jnp.minimum(ref.min(axis=0), comp.min(axis=0)), dtype=EVENT_DTYPE
    )
    hi: NDArray[np.single] = np.asarray(
        jnp.maximum(ref.max(axis=0), comp.max(axis=0)), dtype=EVENT_DTYPE
    )
    edges: NDArray[Any] = np.linspace(start=lo, stop=hi, num=n_bins + 1, axis=-1)
    return edges.astype(EVENT_DTYPE)


def _counts(x: JaxArray, edges: JaxArray, weights: JaxArray) -> JaxArray:
    """Weighted bin counts per column, `(dim, n_bins)`.

    `searchsorted(..., "right") - 1` is what `np.histogram` does with explicit
    edges; the clip is its closed last bin, which is where the maxima land.

    The weights are **centered before they are scattered**, and the mean added
    back through the exact count. Scattering them raw sums ~200 values of
    magnitude ~1 per bin in float32, which is precisely what the `np.histogram`
    this replaces did -- that function accumulates in the weights' own dtype,
    and RAN's weights are float32. Centering leaves the scatter summing
    residuals instead of magnitudes, an order of magnitude smaller, while the
    integer count it is added back to is exact in float32 out to 2**24, far
    above any sample this runs on. Measured against float64, that lands the JS
    divergence within 9e-9 where the old path was 5.9e-7 off -- the difference
    between reaching the sixth decimal `metrics.json` prints and not.

    The mean is itself a float32 reduction and carries its own error, which
    does not matter: it multiplies every bin of the column by the same factor,
    and `_normalize` divides it straight back out.
    """
    n_bins: int = edges.shape[1] - 1
    mean_weight: JaxArray = jnp.mean(weights)
    residuals: JaxArray = weights - mean_weight

    def one_column(col: JaxArray, col_edges: JaxArray) -> JaxArray:
        index: JaxArray = jnp.clip(
            jnp.searchsorted(col_edges, col, side="right") - 1, 0, n_bins - 1
        )
        empty: JaxArray = jnp.zeros((n_bins,), dtype=EVENT_DTYPE)
        count: JaxArray = empty.at[index].add(jnp.ones_like(residuals))
        residual: JaxArray = empty.at[index].add(residuals)
        return count * mean_weight + residual

    return jax.vmap(one_column, in_axes=(1, 0))(x, edges)


def _cdf_gap_integral(ref: JaxArray, comp: JaxArray, weights: JaxArray) -> JaxArray:
    """`integral |F_ref - F_comp| dt` per column: the 1D Wasserstein-1 distance.

    Sort the pooled values, accumulate the two sides' normalized weights into a
    step CDF, integrate the gap -- the estimator
    `scipy.stats.wasserstein_distance` defines, vectorized over columns so one
    dispatch does every dimension. Named as the definition, not as a call: the
    agreement is asserted against scipy in `tests/test_evaluate_metrics.py`,
    which is the only place in this package that still imports it.

    The two CDFs are *never* accumulated separately. Each climbs to 1 while
    their difference stays at the order of the distance being measured, so
    subtracting them afterwards cancels away most of a float32 mantissa.
    Cumulatively summing the signed weights instead keeps the running value at
    the size of the answer, which makes the float32 error relative to it rather
    than to 1 -- and costs one scan instead of two.
    """
    n: int = ref.shape[0]
    signed: JaxArray = jnp.concatenate(
        [
            jnp.full((n,), 1.0 / n, dtype=EVENT_DTYPE),
            -weights / jnp.sum(weights),
        ]
    )
    pooled: JaxArray = jnp.concatenate([ref, comp], axis=0)
    order: JaxArray = jnp.argsort(pooled, axis=0)
    values: JaxArray = jnp.take_along_axis(pooled, order, axis=0)
    gap: JaxArray = jnp.cumsum(
        jnp.take_along_axis(
            jnp.broadcast_to(signed[:, None], pooled.shape), order, axis=0
        ),
        axis=0,
    )
    return jnp.sum(jnp.abs(gap[:-1]) * jnp.diff(values, axis=0), axis=0)


@jax.jit
def _histogram_kernel(
    ref: JaxArray, comp: JaxArray, weights: JaxArray, edges: JaxArray
) -> tuple[JaxArray, JaxArray]:
    ones: JaxArray = jnp.ones((ref.shape[0],), dtype=EVENT_DTYPE)
    return _counts(ref, edges, ones), _counts(comp, edges, weights)


@jax.jit
def _metrics_kernel(
    ref: JaxArray, comp: JaxArray, weights: JaxArray, edges: JaxArray
) -> tuple[JaxArray, JaxArray, JaxArray]:
    ones: JaxArray = jnp.ones((ref.shape[0],), dtype=EVENT_DTYPE)
    return (
        _cdf_gap_integral(ref, comp, weights),
        _counts(ref, edges, ones),
        _counts(comp, edges, weights),
    )


def _normalize(counts: JaxArray) -> NDArray[np.double]:
    """Bin counts to probability masses, on the host in float64.

    The counts come back from device float32 because that is what the scatter
    that produced them sums in; everything downstream of here is a divergence
    of a `dim x n_bins` array, which is free, so it is taken in float64 --
    scores are not pinned to the data's precision. An all-zero histogram is
    left unnormalized rather than divided by zero.
    """
    dense: NDArray[np.double] = np.asarray(counts, dtype=np.double)
    total: NDArray[np.double] = dense.sum(axis=1, keepdims=True)
    return cast("NDArray[np.double]", dense / np.where(total > 0, total, 1.0))


def _wd_per_dim(
    ref: EventArray,
    comp: EventArray,
    weights: EventArray | JaxArray | None = None,
) -> NDArray[np.double]:
    """1D Wasserstein distance per dimension."""
    ref_2d, comp_2d, w = _prepare(ref, comp, weights)
    return np.asarray(_cdf_gap_integral(ref_2d, comp_2d, w), dtype=np.double)


def _normalized_histograms(
    ref: EventArray,
    comp: EventArray,
    weights: EventArray | JaxArray | None = None,
    n_bins: int = 100,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """The `(p, q)` probability histograms, `(dim, n_bins)` each.

    Both share one binning per dimension, which is what makes the divergences
    below comparable across dimensions. `weights` reweights `comp` only.
    """
    ref_2d, comp_2d, w = _prepare(ref, comp, weights)
    edges: JaxArray = jnp.asarray(_bin_edges(ref_2d, comp_2d, n_bins))
    h_ref, h_comp = _histogram_kernel(ref_2d, comp_2d, w, edges)
    return _normalize(h_ref), _normalize(h_comp)


def _relative_entropy(
    x: NDArray[np.double], m: NDArray[np.double]
) -> NDArray[np.double]:
    """`sum_i x_i log(x_i / m_i)` per row, with `scipy.special.rel_entr`'s zeros.

    `rel_entr(0, m)` is 0, not `0 * log(0/m)`. Here `m` is a mean of `x` and
    another distribution, so it can only vanish where `x` does too, and the
    mask that supplies the 0 also keeps the division away from it.
    """
    positive: NDArray[np.bool] = x > 0
    ratio: NDArray[np.double] = np.divide(x, m, out=np.ones_like(x), where=positive)
    return np.sum(np.where(positive, x * np.log(ratio), 0.0), axis=1)


def _js_from_histograms(
    p: NDArray[np.double], q: NDArray[np.double]
) -> NDArray[np.double]:
    """Jensen-Shannon divergence per row: the square of the JS *distance*.

    `(D(p || m) + D(q || m)) / 2` with `m` the midpoint mixture, in nats --
    what `scipy.spatial.distance.jensenshannon(p, q) ** 2` returns, which
    `tests/test_evaluate_metrics.py` asserts against rather than this module
    importing it. Vectorized over rows, because scipy's own `axis=` argument is
    missing from its type stubs and calling it row by row to get around that
    was the last thing here still reaching for scipy at all.

    Each row is divided by its own sum, exactly as scipy does, and that is the
    whole reason the division is not skipped as redundant: `_normalize` leaves
    a histogram with no mass unnormalized, and dividing it by a zero total is
    what turns it into the NaN an empty distribution deserves. A 0 there would
    read as "these two agree perfectly".
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p_norm: NDArray[np.double] = p / p.sum(axis=1, keepdims=True)
        q_norm: NDArray[np.double] = q / q.sum(axis=1, keepdims=True)
    m: NDArray[np.double] = (p_norm + q_norm) / 2
    return (_relative_entropy(p_norm, m) + _relative_entropy(q_norm, m)) / 2


def _triangular_from_histograms(
    p: NDArray[np.double], q: NDArray[np.double]
) -> NDArray[np.double]:
    """Triangular discriminator (Vincze-LeCam divergence) per dimension.

    Delta(p,q) = sum (p_i - q_i)^2 / (p_i + q_i)  x  1e3

    The bin-width factor cancels analytically, so this works directly on
    normalized histograms.
    """
    denom: NDArray[np.double] = p + q
    nonempty: NDArray[np.bool] = denom > 0
    diff: NDArray[np.double] = p - q
    return (
        np.sum(
            np.where(nonempty, diff**2 / np.where(nonempty, denom, 1.0), 0.0), axis=1
        )
        * 1e3
    )


def _js_per_dim(
    ref: EventArray,
    comp: EventArray,
    weights: EventArray | JaxArray | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    return _js_from_histograms(*_normalized_histograms(ref, comp, weights, n_bins))


def _triangular_per_dim(
    ref: EventArray,
    comp: EventArray,
    weights: EventArray | JaxArray | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    return _triangular_from_histograms(
        *_normalized_histograms(ref, comp, weights, n_bins)
    )


class MetricSet(NamedTuple):
    """Every metric `metrics.json` records, one entry per dimension."""

    wasserstein: NDArray[np.double]
    jensenshannon: NDArray[np.double]
    triangular: NDArray[np.double]


def _metrics_per_dim(
    ref: EventArray,
    comp: EventArray,
    weights: EventArray | JaxArray | None = None,
    n_bins: int = 100,
) -> MetricSet:
    """All three metrics in one device pass.

    The two divergences read the *same* pair of histograms. Called through
    `_js_per_dim` and `_triangular_per_dim` they would each build their own,
    which is two identical scatters over the full sample for two reductions
    over `dim x n_bins` values -- the reason `evaluate_run` goes through here
    and the single-metric helpers stay for the callers that want one number.
    """
    ref_2d, comp_2d, w = _prepare(ref, comp, weights)
    edges: JaxArray = jnp.asarray(_bin_edges(ref_2d, comp_2d, n_bins))
    distance, h_ref, h_comp = _metrics_kernel(ref_2d, comp_2d, w, edges)
    p: NDArray[np.double] = _normalize(h_ref)
    q: NDArray[np.double] = _normalize(h_comp)
    return MetricSet(
        wasserstein=np.asarray(distance, dtype=np.double),
        jensenshannon=_js_from_histograms(p, q),
        triangular=_triangular_from_histograms(p, q),
    )


def _metric_entry(before: MetricSet, after: MetricSet, index: int) -> dict[str, float]:
    """One variable's row of `metrics.json`, in `MetricSet` field order.

    Every value goes through `float()`. These arrive as `np.double`, which
    subclasses Python `float` and would serialize either way -- but that is the
    property that let a `np.float32` reach `json.dump` unnoticed elsewhere in
    the pipeline, and coercing here means the writer does not depend on which
    one a metric happens to return.
    """
    entry: dict[str, float] = {}
    for name, was_all, now_all in zip(MetricSet._fields, before, after, strict=True):
        was: float = float(was_all[index])
        now: float = float(now_all[index])
        entry[f"{name}_before"] = was
        entry[f"{name}_after"] = now
        entry[f"{name}_improvement_pct"] = _improvement(was, now)
    return entry


def _improvement(before: float, after: float) -> float:
    return (1 - after / before) * 100 if before > 0 else 0.0


def evaluate_run(run_dir: Path, force: bool = False) -> dict[str, Any]:
    """Evaluate a single run directory."""
    out_path: Path = run_dir / "metrics.json"

    if out_path.exists() and not force:
        logger.info("%s: metrics.json exists, skipping (use --force)", run_dir.name)
        return cast("dict[str, Any]", json.loads(out_path.read_text()))

    # Imported here, not at module scope, so this module stays keras-free on
    # import.
    import keras

    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    logger.info("%s: loading model and data...", run_dir.name)
    g: RANModel = keras.saving.load_model(run_dir / "generator.keras")

    splits: DatasetSplits = _load_splits(config)
    test: Populations = _collect_test_data(splits.test).partition()
    # Left on device: every metric below runs there, so the only array that
    # crosses back is the handful of numbers per dimension they reduce to.
    w: JaxArray = _generator_weights(g, test.mc.z)

    # Variable names for labeling
    dataset: str = config.get("dataset", "gaussian")
    dim: int = config["dim"]
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(dim)]

    metrics: dict[str, Any] = {}

    for level, data, mc in [
        ("detector", test.data, test.mc.x),
        ("particle", test.require_truth(), test.mc.z),
    ]:
        before: MetricSet = _metrics_per_dim(data, mc)
        after: MetricSet = _metrics_per_dim(data, mc, weights=w)

        for i, var in enumerate(var_names):
            metrics[f"{level}_{var}"] = _metric_entry(before, after, i)

    json.dump(obj=metrics, fp=out_path.open("w"), indent=2)
    logger.info("%s: saved metrics to %s", run_dir.name, out_path)
    render_metrics(run_dir.name, metrics, var_names)
    return metrics


def render_metrics(
    run_name: str,
    metrics: dict[str, Any],
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
