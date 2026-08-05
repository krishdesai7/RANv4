from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from ran.evaluate import (
    _improvement,
    _js_per_dim,
    _load_splits,
    _triangular_per_dim,
    _wd_per_dim,
    apply_to_runs,
    render_metrics,
)
from ran.train import EPS

if TYPE_CHECKING:
    from typing import Any, Final, Literal

    from numpy.typing import NDArray

    from ran.data import ArrayDataset, DatasetSplits

logger: logging.Logger = logging.getLogger(__name__)


DEFAULT_PURITY_THRESHOLD: Final[np.double] = np.sqrt(0.5, dtype=np.double)


class MetricRecord(TypedDict):
    wasserstein_before: float
    wasserstein_after: float
    wasserstein_improvement_pct: float
    jensenshannon_before: float
    jensenshannon_after: float
    jensenshannon_improvement_pct: float
    triangular_before: float
    triangular_after: float
    triangular_improvement_pct: float


@dataclass(frozen=True)
class IBUConfig:
    source: dict[str, Any]
    dataset: Literal["gaussian", "jets"]
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]


@dataclass(frozen=True)
class VariableOutcome:
    variable_name: str
    status: Literal["completed", "skipped"]
    n_bins: int
    skip_reason: str | None = None


@dataclass(frozen=True)
class IBUResult:
    metrics: dict[str, MetricRecord]
    variable_names: tuple[str, ...]
    weights: NDArray[np.double]
    outcomes: tuple[VariableOutcome, ...]


@dataclass(frozen=True)
class _IBUData:
    response_gen: NDArray[np.double]
    response_sim: NDArray[np.double]
    observed_reco: NDArray[np.double]
    test_data_gen: NDArray[np.double]
    test_data_reco: NDArray[np.double]
    test_mc_gen: NDArray[np.double]
    test_mc_reco: NDArray[np.double]


@dataclass(frozen=True)
class _VariableUnfolding:
    weights: NDArray[np.double]
    outcome: VariableOutcome


def _positive_int(value: object, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _parse_dataset(raw: dict[str, Any]) -> Literal["gaussian", "jets"]:
    dataset: object = raw.get("dataset", "gaussian")
    if dataset == "gaussian":
        return "gaussian"
    if dataset == "jets":
        return "jets"
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _parse_variable_names(
    raw: dict[str, Any], dataset: Literal["gaussian", "jets"], dim: int
) -> tuple[str, ...]:
    if dataset == "gaussian":
        return tuple(f"dim_{i}" for i in range(dim))

    variables: object = raw.get("variables")
    if not isinstance(variables, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in variables
    ):
        raise ValueError("variables must be a sequence of nonempty strings")
    variable_names = tuple(variables)
    if len(variable_names) != dim:
        raise ValueError(
            f"variables has length {len(variable_names)}, expected dim={dim}"
        )
    return variable_names


def _parse_config(raw: object) -> IBUConfig:
    if not isinstance(raw, dict):
        raise ValueError("IBU config must be a JSON object")  # ruff: ignore[type-check-without-type-error]

    dim: int = _positive_int(raw.get("dim"), "dim")
    n_samples: int = _positive_int(raw.get("n_samples"), "n_samples")
    batch_size: int = _positive_int(raw.get("batch_size"), "batch_size")
    data_seed: object = raw.get("data_seed", 42)
    if type(data_seed) is not int:
        raise ValueError("data_seed must be an integer")

    dataset = _parse_dataset(raw)
    variable_names = _parse_variable_names(raw, dataset, dim)

    return IBUConfig(
        source=dict(raw),
        dataset=dataset,
        dim=dim,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=data_seed,
        variable_names=variable_names,
    )


def _validated_arrays(
    split: ArrayDataset, expected_dim: int
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]:
    z, x, y = split.as_arrays()
    if z.ndim != 2 or x.ndim != 2 or z.shape != x.shape:
        raise ValueError("z and x must be identically shaped two-dimensional arrays")
    if z.shape[1] != expected_dim:
        raise ValueError(f"array dimension {z.shape[1]}, expected dim={expected_dim}")
    if y.ndim != 1 or y.shape[0] != z.shape[0]:
        raise ValueError("y must be one-dimensional with one label per row")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(x)):
        raise ValueError("z and x values must be finite")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be only zero or one")
    return z, x, y


def _prepare_data(splits: DatasetSplits, expected_dim: int) -> _IBUData:
    """Validate dataset arrays and separate response and evaluation populations."""
    arrays = [
        _validated_arrays(split, expected_dim)
        for split in (splits.train, splits.val, splits.test)
    ]

    z_all: NDArray[np.double] = np.concatenate([item[0] for item in arrays], axis=0)
    x_all: NDArray[np.double] = np.concatenate([item[1] for item in arrays], axis=0)
    y_all: NDArray[np.ubyte] = np.concatenate([item[2] for item in arrays], axis=0)
    z_test, x_test, y_test = arrays[-1]

    response_mask = y_all == 0
    observed_mask = y_all == 1
    test_mc_mask = y_test == 0
    test_data_mask = y_test == 1
    if not np.any(response_mask):
        raise ValueError("response MC population must not be empty")
    if not np.any(observed_mask):
        raise ValueError("observed data population must not be empty")
    if not np.any(test_mc_mask):
        raise ValueError("test MC population must not be empty")
    if not np.any(test_data_mask):
        raise ValueError("test data population must not be empty")

    return _IBUData(
        response_gen=z_all[response_mask],
        response_sim=x_all[response_mask],
        observed_reco=x_all[observed_mask],
        test_data_gen=z_test[test_data_mask],
        test_data_reco=x_test[test_data_mask],
        test_mc_gen=z_test[test_mc_mask],
        test_mc_reco=x_test[test_mc_mask],
    )


def _assign_bins(
    values: NDArray[np.double], edges: NDArray[np.double]
) -> NDArray[np.intp]:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("bin values must be a finite one-dimensional array")
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bin edges must be a strictly increasing 1D array")
    n_bins = edges.size - 1
    return (np.clip(np.digitize(values, edges), 1, n_bins) - 1).astype(
        np.intp, copy=False
    )


def _bin_counts(indices: NDArray[np.intp], n_bins: int) -> NDArray[np.double]:
    if n_bins < 1 or indices.ndim != 1:
        raise ValueError("bin indices must be one-dimensional with n_bins >= 1")
    if np.any((indices < 0) | (indices >= n_bins)):
        raise ValueError("bin index outside configured range")
    return np.bincount(indices, minlength=n_bins).astype(np.double)


def _unfolded_to_bin_weights(
    unfolded: NDArray[np.double], prior: NDArray[np.double]
) -> NDArray[np.double]:
    if (
        unfolded.ndim != 1
        or prior.ndim != 1
        or unfolded.size == 0
        or unfolded.shape != prior.shape
    ):
        raise ValueError(
            "unfolded and prior must have matching nonempty one-dimensional shapes"
        )
    if not np.all(np.isfinite(unfolded)) or not np.all(np.isfinite(prior)):
        raise ValueError("unfolded and prior must be finite")
    if np.any(unfolded < 0) or np.any(prior < 0):
        raise ValueError("unfolded and prior must be nonnegative")
    if np.any((prior == 0) & (unfolded > EPS)):
        raise ValueError("unfolded mass in a zero-prior bin")

    weights: NDArray[np.double] = np.zeros_like(unfolded, dtype=np.double)
    np.divide(unfolded, prior, out=weights, where=prior > 0)
    return weights


def _normalize_weights(weights: NDArray[np.double]) -> NDArray[np.double]:
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")

    mean: np.double = np.mean(weights, dtype=np.double)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("weights mean must be finite and strictly positive")

    normalized: NDArray[np.double] = np.asarray(weights / mean, dtype=np.double)
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError("normalized weights must be finite and nonnegative")
    if not np.isclose(normalized.mean(), 1.0):
        raise ValueError("normalized weights must have mean one")
    return normalized


def _next_pure_edge(
    gen_sorted: NDArray[np.double],
    upper_sorted: NDArray[np.double],
    lower_by_upper: NDArray[np.double],
    lo: np.double,
    gen_max: np.double,
    purity_threshold: np.double,
    n_candidates: int = 100,
) -> np.double | None:
    """Return the first candidate edge whose bin exceeds the purity threshold.

    `gen_sorted` is sorted `gen`.

    `upper_sorted` is sorted `maximum(gen, sim)`, and
    `lower_by_upper` is `minimum(gen, sim)` in the corresponding order.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")

    candidates: NDArray[np.double] = np.linspace(
        lo + 1 / n_candidates,
        gen_max,
        n_candidates,
        dtype=gen_sorted.dtype,
    )

    # Denominator = count(lo <= gen < candidate) for every candidate.
    truth_start: np.intp = np.searchsorted(gen_sorted, lo, side="left")
    truth_stop: NDArray[np.intp] = np.searchsorted(gen_sorted, candidates, side="left")
    n_truth: NDArray[np.intp] = truth_stop - truth_start

    # Because upper_sorted is ordered, the first k elements are precisely
    # those for which max(gen, reco) < candidates[j].
    upper_stop: NDArray[np.intp] = np.searchsorted(
        upper_sorted, candidates, side="left"
    )

    # Among those elements, count the ones satisfying
    # min(gen, reco) >= lo.
    prefix: NDArray[np.ulong] = np.empty(lower_by_upper.size + 1, dtype=np.ulong)
    prefix[0] = 0
    np.cumsum(
        lower_by_upper >= lo,
        dtype=np.ulong,
        out=prefix[1:],
    )
    n_both: NDArray[np.ulong] = prefix[upper_stop]

    purity: NDArray[np.double] = np.zeros(n_candidates, dtype=np.double)
    np.divide(
        n_both,
        n_truth,
        out=purity,
        where=n_truth != 0,
    )

    qualifying: NDArray[np.intp] = np.flatnonzero(
        (n_truth != 0) & (purity > purity_threshold)
    )
    if qualifying.size == 0:
        return None

    return candidates[qualifying[0]]


def _purity_bins(
    gen: NDArray[np.double],
    sim: NDArray[np.double],
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
    max_bins: int = 50,
) -> NDArray[np.double]:
    """Determine bin edges where purity exceeds the threshold."""
    if gen.ndim != 1 or sim.ndim != 1:
        raise ValueError("gen and sim must be one-dimensional")
    if gen.shape != sim.shape:
        raise ValueError("gen and sim must have the same shape")
    if gen.size == 0:
        raise ValueError("gen and sim must not be empty")
    if max_bins <= 0:
        raise ValueError("max_bins must be positive")

    # One-time preprocessing.
    gen_sorted: NDArray[np.double] = np.sort(gen)

    lower: NDArray[np.double] = np.minimum(gen, sim)
    upper: NDArray[np.double] = np.maximum(gen, sim)

    upper_order: NDArray[np.intp] = np.argsort(upper)
    upper_sorted: NDArray[np.double] = upper[upper_order]
    lower_by_upper: NDArray[np.double] = lower[upper_order]

    # max_bins bins require at most max_bins + 1 edges.
    edges: NDArray[np.double] = np.empty(max_bins + 1, dtype=gen.dtype)
    edges[0] = gen.min()
    n_edges = 1

    gen_max: Final[np.double] = gen.max()
    while n_edges <= max_bins and edges[n_edges - 1] < gen_max:
        edge: np.double | None = _next_pure_edge(
            gen_sorted=gen_sorted,
            upper_sorted=upper_sorted,
            lower_by_upper=lower_by_upper,
            lo=edges[n_edges - 1],
            gen_max=gen_max,
            purity_threshold=purity_threshold,
        )
        if edge is None:
            break

        edges[n_edges] = edge
        n_edges += 1

    return edges[:n_edges]


def _build_response(
    gen_bins: NDArray[np.long],
    reco_bins: NDArray[np.long],
    n_bins: int,
) -> NDArray[np.double]:
    """Build row-normalized response matrix R[t,r] = P(reco=r | truth=t)."""
    response: NDArray[np.double] = np.zeros((n_bins, n_bins), dtype=np.double)
    np.add.at(response, (gen_bins, reco_bins), 1)
    row_sums: NDArray[np.double] = response.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    response /= row_sums
    return response


def _ibu(
    prior: NDArray[np.double],
    data_hist: NDArray[np.double],
    response: NDArray[np.double],
    n_iterations: int,
) -> NDArray[np.double]:
    """Iterative Bayesian Unfolding.

    Args:
        prior: Initial truth estimate (MC gen histogram), shape (n_bins,).
        data_hist: Observed reco-level measured histogram, shape (n_bins,).
        response: R[t,r] = P(sim=r | gen=t), shape (n_bins, n_bins).
        n_iterations: Number of unfolding iterations.

    Returns:
        Unfolded truth histogram, shape (n_bins,).
    """
    posterior: NDArray[np.double] = prior.copy()
    m: NDArray[np.double]
    for _ in range(n_iterations):
        # Bayes: P(t|r) = R[t,r]*P(t) / sum_t' R[t',r]*P(t')
        m = response.T * posterior  # m[r,t] = R[t,r] * P(t)
        m /= m.sum(axis=1, keepdims=True) + EPS  # m[r,t] = P(t|r)
        posterior = m.T @ data_hist  # P(t) = sum_r P(t|r) * data(r)
    return posterior


def _unfold_variable(
    variable_name: str,
    response_gen: NDArray[np.double],
    response_sim: NDArray[np.double],
    observed_reco: NDArray[np.double],
    test_mc_gen: NDArray[np.double],
    n_iterations: int,
    purity_threshold: np.double,
) -> _VariableUnfolding:
    bins: NDArray[np.double] = _purity_bins(
        response_gen, response_sim, purity_threshold
    )
    n_bins: int = bins.size - 1
    if n_bins < 2:
        logger.warning("%s: only %d bin(s), skipping", variable_name, n_bins)
        return _VariableUnfolding(
            weights=np.ones(test_mc_gen.size, dtype=np.double),
            outcome=VariableOutcome(
                variable_name,
                "skipped",
                n_bins,
                "fewer than two purity bins",
            ),
        )

    response_gen_bins: NDArray[np.intp] = _assign_bins(response_gen, bins)
    response_sim_bins: NDArray[np.intp] = _assign_bins(response_sim, bins)
    observed_reco_bins: NDArray[np.intp] = _assign_bins(observed_reco, bins)
    test_mc_gen_bins: NDArray[np.intp] = _assign_bins(test_mc_gen, bins)

    response: NDArray[np.double] = _build_response(
        response_gen_bins, response_sim_bins, n_bins
    )
    prior: NDArray[np.double] = _bin_counts(response_gen_bins, n_bins)
    observed: NDArray[np.double] = _bin_counts(observed_reco_bins, n_bins)
    assert prior.sum() == response_gen.size  # ruff: ignore[assert]
    assert observed.sum() == observed_reco.size  # ruff: ignore[assert]

    unfolded: NDArray[np.double] = _ibu(prior, observed, response, n_iterations)
    bin_weights: NDArray[np.double] = _unfolded_to_bin_weights(unfolded, prior)
    weights: NDArray[np.double] = _normalize_weights(bin_weights[test_mc_gen_bins])
    logger.info("%s: %d bins, %d iterations", variable_name, n_bins, n_iterations)
    return _VariableUnfolding(
        weights=weights,
        outcome=VariableOutcome(variable_name, "completed", n_bins),
    )


def _evaluate_dimension(
    reference: NDArray[np.double],
    comparison: NDArray[np.double],
    weights: NDArray[np.double],
) -> MetricRecord:
    wasserstein_before: float = _wd_per_dim(reference, comparison)[0]
    wasserstein_after: float = _wd_per_dim(reference, comparison, weights=weights)[0]
    jensenshannon_before: float = _js_per_dim(reference, comparison)[0]
    jensenshannon_after: float = _js_per_dim(reference, comparison, weights=weights)[0]
    triangular_before: float = _triangular_per_dim(reference, comparison)[0]
    triangular_after: float = _triangular_per_dim(
        reference, comparison, weights=weights
    )[0]
    return {
        "wasserstein_before": wasserstein_before,
        "wasserstein_after": wasserstein_after,
        "wasserstein_improvement_pct": _improvement(
            wasserstein_before, wasserstein_after
        ),
        "jensenshannon_before": jensenshannon_before,
        "jensenshannon_after": jensenshannon_after,
        "jensenshannon_improvement_pct": _improvement(
            jensenshannon_before, jensenshannon_after
        ),
        "triangular_before": triangular_before,
        "triangular_after": triangular_after,
        "triangular_improvement_pct": _improvement(triangular_before, triangular_after),
    }


def _run_and_evaluate(
    config: IBUConfig,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> IBUResult:
    """Run 1D IBU per variable and evaluate on test set."""
    _positive_int(n_iterations, "n_iterations")
    if not np.isfinite(purity_threshold) or not 0 <= purity_threshold <= 1:
        raise ValueError("purity_threshold must be finite and between zero and one")

    splits = _load_splits(config.source)
    data = _prepare_data(splits, config.dim)
    weights = np.empty((config.dim, data.test_mc_gen.shape[0]), dtype=np.double)
    metrics: dict[str, MetricRecord] = {}
    outcomes: list[VariableOutcome] = []

    for dimension, variable_name in enumerate(config.variable_names):
        unfolding = _unfold_variable(
            variable_name=variable_name,
            response_gen=data.response_gen[:, dimension],
            response_sim=data.response_sim[:, dimension],
            observed_reco=data.observed_reco[:, dimension],
            test_mc_gen=data.test_mc_gen[:, dimension],
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        )
        weights[dimension] = unfolding.weights
        outcomes.append(unfolding.outcome)
        metrics[f"detector_{variable_name}"] = _evaluate_dimension(
            data.test_data_reco[:, dimension],
            data.test_mc_reco[:, dimension],
            unfolding.weights,
        )
        metrics[f"particle_{variable_name}"] = _evaluate_dimension(
            data.test_data_gen[:, dimension],
            data.test_mc_gen[:, dimension],
            unfolding.weights,
        )

    return IBUResult(
        metrics=metrics,
        variable_names=config.variable_names,
        weights=weights,
        outcomes=tuple(outcomes),
    )


def evaluate_single(
    run_dir: str | Path,
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> dict[str, MetricRecord]:
    """Run IBU on a single RAN run's dataset and save comparison metrics."""
    run_dir = Path(run_dir)
    out_path: Path = run_dir / "metrics_ibu.json"

    if out_path.exists() and not force:
        logger.info("%s: metrics_ibu.json exists, skipping (use --force)", run_dir.name)
        return json.loads(out_path.read_text())

    raw_config: object = json.loads((run_dir / "config.json").read_text())
    config = _parse_config(raw_config)
    logger.info(
        "%s: running IBU (niter=%d, purity=%.4f)...",
        run_dir.name,
        n_iterations,
        purity_threshold,
    )
    result = _run_and_evaluate(
        config,
        n_iterations=n_iterations,
        purity_threshold=purity_threshold,
    )

    json.dump(result.metrics, out_path.open("w"), indent=2)
    weights_path: Path = run_dir / "ibu_weights.npz"
    np.savez(
        weights_path,
        # savez is `savez(file, *args, allow_pickle=True, **kwds)`. Our keys are
        # built by f-string, so their type is plain `str` -- one of them *could*
        # be "allow_pickle", which is declared bool. The complaint is therefore
        # sound rather than a stub bug (ty reports it too); it just cannot happen
        # here. A literal-key dict checks clean, but ours cannot be one.
        **{f"weights_{i}": weights for i, weights in enumerate(result.weights)},  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    )
    logger.info(
        "%s: saved IBU metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [IBU]", result.metrics, list(result.variable_names))
    return result.metrics


def evaluate_runs(
    run_dir: str | Path = "runs",
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> None:
    """Run IBU baseline on completed RAN runs.

    Args:
        run_dir: Path to a single run or directory of runs.
        force: Recompute even if metrics_ibu.json exists.
        n_iterations: Number of IBU iterations.
        purity_threshold: Purity threshold for automatic binning.
    """
    apply_to_runs(
        run_dir,
        lambda d: evaluate_single(
            d,
            force=force,
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        ),
        "evaluate with IBU",
        logger,
    )
