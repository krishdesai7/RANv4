from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..evaluate import apply_to_runs, render_metrics
from ..rantypes import DEFAULT_PURITY_THRESHOLD, IBUResult, Populations, VariableOutcome
from ..train import EPS
from ._shared import (
    evaluate_dimension,
    load_populations,
    parse_run_config,
)

if TYPE_CHECKING:
    from logging import Logger
    from typing import Final

    from numpy._typing import _DTypeLikeFloat
    from numpy.typing import NDArray

    from ..rantypes import EventArray, MetricRecord, RunConfig

logger: Logger = logging.getLogger(name=__name__)


@dataclass(frozen=True, eq=False, slots=True)
class _BinnedReweighting:
    edges: EventArray
    bin_weights: EventArray

    def weights_for(self, gen: EventArray) -> EventArray:
        """Per-event weights for particle-level values `gen`, mean one."""
        return _normalize_weights(self.bin_weights[_assign_bins(gen, self.edges)])


@dataclass(frozen=True, eq=False, slots=True)
class VariableUnfolding:
    """One variable's reweighting, or `None` where it could not be fit."""

    reweighting: _BinnedReweighting | None
    outcome: VariableOutcome

    def weights_for(self, gen: EventArray) -> EventArray:
        if self.reweighting is None:
            return np.ones(gen.shape[0], dtype=gen.dtype)
        return self.reweighting.weights_for(gen)


def _assign_bins(values: EventArray, edges: EventArray, /) -> NDArray[np.intp]:
    if values.ndim != 1 or not np.all(a=np.isfinite(values)):
        raise ValueError("bin values must be a finite one-dimensional array")
    if edges.ndim != 1 or edges.size < 2 or not np.all(a=np.diff(a=edges) > 0):
        raise ValueError("bin edges must be a strictly increasing 1D array")
    n_bins: int = edges.size - 1
    return (
        np.clip(np.digitize(x=values, bins=edges), a_min=1, a_max=n_bins) - 1
    ).astype(dtype=np.intp, copy=False)


def _bin_counts(indices: NDArray[np.intp], n_bins: int, /) -> NDArray[np.intp]:
    if n_bins < 1 or indices.ndim != 1:
        raise ValueError("bin indices must be one-dimensional with n_bins >= 1")
    if np.any(a=(indices < 0) | (indices >= n_bins)):
        raise ValueError("bin index outside configured range")
    return np.bincount(indices, minlength=n_bins)


def _unfolded_to_bin_weights(unfolded: EventArray, prior: EventArray) -> EventArray:
    if (
        unfolded.ndim != 1
        or prior.ndim != 1
        or unfolded.size == 0
        or unfolded.shape != prior.shape
    ):
        raise ValueError(
            "unfolded and prior must have matching nonempty one-dimensional shapes"
        )
    if not np.all(a=np.isfinite(unfolded)) or not np.all(a=np.isfinite(prior)):
        raise ValueError("unfolded and prior must be finite")
    if np.any(a=unfolded < 0) or np.any(a=prior < 0):
        raise ValueError("unfolded and prior must be nonnegative")
    if np.any(a=(prior == 0) & (unfolded > EPS)):
        raise ValueError("unfolded mass in a zero-prior bin")

    weights: EventArray = np.zeros_like(a=unfolded)
    np.divide(unfolded, prior, out=weights, where=prior > 0)
    return weights


def _normalize_weights(weights: EventArray) -> EventArray:
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional vector")
    if not np.all(a=np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(a=weights < 0):
        raise ValueError("weights must be nonnegative")

    mean: np.single = np.mean(a=weights)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("weights mean must be finite and strictly positive")

    normalized: EventArray = np.divide(weights, mean)
    if not np.all(a=np.isfinite(normalized)) or np.any(a=normalized < 0):
        raise ValueError("normalized weights must be finite and nonnegative")
    # The division runs in the caller's precision; only the check that it
    # worked is accumulated wider, so a float32 resummation cannot fail it.
    if not np.isclose(a=normalized.mean(dtype=np.double), b=1.0):
        raise ValueError("normalized weights must have mean one")
    return normalized


def _next_pure_edge(
    gen_sorted: EventArray,
    upper_sorted: EventArray,
    lower_by_upper: EventArray,
    lo: np.single,
    gen_max: np.single,
    purity_threshold: float,
    n_candidates: int = 100,
) -> np.single | None:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")

    candidates: EventArray = np.linspace(
        start=lo + 1 / n_candidates,
        stop=gen_max,
        num=n_candidates,
        dtype=gen_sorted.dtype,
    )

    # Denominator = count(lo <= gen < candidate) for every candidate.
    truth_start: np.intp = np.searchsorted(a=gen_sorted, v=lo, side="left")
    truth_stop: NDArray[np.intp] = np.searchsorted(
        a=gen_sorted, v=candidates, side="left"
    )
    n_truth: NDArray[np.intp] = truth_stop - truth_start

    # Because upper_sorted is ordered, the first k elements are precisely
    # those for which max(gen, reco) < candidates[j].
    upper_stop: NDArray[np.intp] = np.searchsorted(
        a=upper_sorted, v=candidates, side="left"
    )

    # Among those elements, count the ones satisfying min(gen, reco) >= lo.
    prefix: NDArray[np.ulong] = np.empty(shape=lower_by_upper.size + 1, dtype=np.ulong)
    prefix[0] = 0
    np.cumsum(
        a=lower_by_upper >= lo,
        dtype=np.ulong,
        out=prefix[1:],
    )
    n_both: NDArray[np.ulong] = prefix[upper_stop]

    purity: NDArray[np.double] = np.zeros(shape=n_candidates, dtype=np.double)
    np.divide(
        n_both,
        n_truth,
        out=purity,
        where=n_truth != 0,
    )

    qualifying: NDArray[np.intp] = np.flatnonzero(
        a=(n_truth != 0) & (purity > purity_threshold)
    )
    if qualifying.size == 0:
        return None

    return candidates[qualifying[0]]


def _purity_bins(
    gen: EventArray,
    sim: EventArray,
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
    max_bins: int = 50,
) -> EventArray:
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
    gen_sorted: EventArray = np.sort(a=gen)

    lower: EventArray = np.minimum(gen, sim)
    upper: EventArray = np.maximum(gen, sim)

    upper_order: NDArray[np.intp] = np.argsort(a=upper)
    upper_sorted: EventArray = upper[upper_order]
    lower_by_upper: EventArray = lower[upper_order]

    # max_bins bins require at most max_bins + 1 edges.
    edges: EventArray = np.empty(shape=max_bins + 1, dtype=gen.dtype)
    edges[0] = gen.min()
    n_edges = 1

    gen_max: Final[np.single] = gen.max()
    while n_edges <= max_bins and edges[n_edges - 1] < gen_max:
        edge: np.single | None = _next_pure_edge(
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
    gen_bins: NDArray[np.intp],
    sim_bins: NDArray[np.intp],
    n_bins: int,
    dtype: _DTypeLikeFloat,
    /,
) -> EventArray:
    """Build row-normalized response matrix R[t,r] = P(reco=r | truth=t)."""
    response: EventArray = np.zeros(shape=(n_bins, n_bins), dtype=dtype)
    np.add.at(response, (gen_bins, sim_bins), 1)
    row_sums: EventArray = response.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    response /= row_sums
    return response


def _ibu(
    prior: EventArray,
    data_hist: EventArray,
    response: EventArray,
    n_iterations: int,
    strict: bool = False,
) -> EventArray:

    posterior: EventArray = prior.copy()

    for _ in range(n_iterations):
        # The NumPy-stub loses the specific floating precision
        # There is no type promotion at runtime
        marginal: EventArray = response.T @ posterior  # pyrefly: ignore[bad-assignment]
        if strict and np.any(a=(marginal == 0) & (data_hist != 0)):
            raise ValueError(
                "Observed data has zero support under the response and prior"
            )
        # `out=` makes the value of the skipped entries zero.
        likelihood: EventArray = np.zeros_like(a=posterior)
        np.divide(
            data_hist,
            marginal,
            out=likelihood,
            where=marginal != 0,
        )
        posterior *= response @ likelihood
    return posterior


def unfold_variable(
    variable_name: str,
    mc_gen: EventArray,
    mc_sim: EventArray,
    observed: EventArray,
    n_iterations: int,
    purity_threshold: float,
) -> VariableUnfolding:
    dtype: np.dtype[np.single] = mc_gen.dtype
    bins: EventArray = _purity_bins(mc_gen, mc_sim, purity_threshold)
    n_bins: int = bins.size - 1
    if n_bins < 2:
        logger.warning("%s: only %d bin(s), skipping", variable_name, n_bins)
        return VariableUnfolding(
            reweighting=None,
            outcome=VariableOutcome(
                variable_name,
                "skipped",
                n_bins,
                skip_reason="fewer than two purity bins",
            ),
        )

    mc_gen_bins: NDArray[np.intp] = _assign_bins(mc_gen, bins)
    mc_sim_bins: NDArray[np.intp] = _assign_bins(mc_sim, bins)
    observed_bins: NDArray[np.intp] = _assign_bins(observed, bins)

    response: EventArray = _build_response(mc_gen_bins, mc_sim_bins, n_bins, dtype)
    prior: EventArray = _bin_counts(mc_gen_bins, n_bins).astype(dtype)
    data_hist: EventArray = _bin_counts(observed_bins, n_bins).astype(dtype)
    # Accumulated in float64 whatever the unfolding runs in: these are exact
    # integer counts, and float32 stops representing those past 2**24, which
    # would fail the comparison on sample size alone.
    prior_count: np.double = prior.sum(dtype=np.double)
    if prior_count != mc_gen.size:
        raise ValueError(
            "prior/response population count mismatch: "
            f"actual={prior_count}, expected={mc_gen.size}"
        )
    observed_count: np.double = data_hist.sum(dtype=np.double)
    if observed_count != observed.size:
        raise ValueError(
            "observed/data population count mismatch: "
            f"actual={observed_count}, expected={observed.size}"
        )

    unfolded: EventArray = _ibu(prior, data_hist, response, n_iterations)
    logger.info("%s: %d bins, %d iterations", variable_name, n_bins, n_iterations)
    return VariableUnfolding(
        reweighting=_BinnedReweighting(
            edges=bins,
            bin_weights=_unfolded_to_bin_weights(unfolded, prior),
        ),
        outcome=VariableOutcome(variable_name, "completed", n_bins),
    )


def _run_and_evaluate(
    config: RunConfig,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> IBUResult:
    """Run 1D IBU per variable and evaluate on test set."""
    if type(n_iterations) is not int or n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer")
    if not np.isfinite(purity_threshold) or not 0 <= purity_threshold <= 1:
        raise ValueError("purity_threshold must be finite and between zero and one")

    full: Populations
    test: Populations
    full, test = load_populations(config)
    test_truth: NDArray[np.single] = test.require_truth()
    weights: NDArray[np.single] = np.empty(
        shape=(config.dim, len(test.mc)), dtype=np.single
    )
    metrics: dict[str, MetricRecord] = {}
    outcomes: list[VariableOutcome] = []

    for dimension, variable_name in enumerate(iterable=config.variable_names):
        # Fit on every split, then score the test split with the result.
        unfolding: VariableUnfolding = unfold_variable(
            variable_name=variable_name,
            mc_gen=full.mc.z[:, dimension],
            mc_sim=full.mc.x[:, dimension],
            observed=full.data[:, dimension],
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        )
        test_weights: NDArray[np.single] = unfolding.weights_for(
            gen=test.mc.z[:, dimension]
        )
        weights[dimension] = test_weights
        outcomes.append(unfolding.outcome)
        metrics[f"detector_{variable_name}"] = evaluate_dimension(
            reference=test.data[:, dimension],
            comparison=test.mc.x[:, dimension],
            weights=test_weights,
        )
        metrics[f"particle_{variable_name}"] = evaluate_dimension(
            reference=test_truth[:, dimension],
            comparison=test.mc.z[:, dimension],
            weights=test_weights,
        )

    return IBUResult(
        metrics=metrics,
        variable_names=config.variable_names,
        weights=weights,
        outcomes=tuple(outcomes),
    )


def evaluate_single(
    run_dir: Path,
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> dict[str, MetricRecord]:
    """Run IBU on a single run's dataset and save comparison metrics."""
    out_path: Path = run_dir / "metrics_ibu.json"

    if out_path.exists() and not force:
        logger.info("%s: metrics_ibu.json exists, skipping (use --force)", run_dir.name)
        return json.loads(s=out_path.read_text())

    raw_config: object = json.loads(s=(run_dir / "config.json").read_text())
    config: RunConfig = parse_run_config(raw_config)
    logger.info(
        "%s: running IBU (niter=%d, purity=%.4f)...",
        run_dir.name,
        n_iterations,
        purity_threshold,
    )
    result: IBUResult = _run_and_evaluate(
        config,
        n_iterations=n_iterations,
        purity_threshold=purity_threshold,
    )

    json.dump(obj=result.metrics, fp=out_path.open(mode="w"), indent=2)
    weights_path: Path = run_dir / "ibu_weights.npz"
    np.savez(
        weights_path,
        # savez is `savez(file, *args, allow_pickle:bool=True, **kwds)`. The keys are
        # built by f-string, so their type is plain `str`.
        **{
            f"weights_{i}": weights for i, weights in enumerate(iterable=result.weights)
        },  # pyrefly: ignore[bad-argument-type]
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
    run_dir: Path = Path("runs"),
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> None:
    apply_to_runs(
        run_dir,
        evaluate_one=lambda run_dir: evaluate_single(
            run_dir,
            force,
            n_iterations,
            purity_threshold,
        ),
        description="evaluate with IBU",
        log=logger,
    )
