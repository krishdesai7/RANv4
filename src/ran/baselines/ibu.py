from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ran.evaluate import (
    _collect_test_data,
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
    from typing import Any, Final

    from numpy.typing import NDArray

    from ran.data import DatasetSplits

logger: logging.Logger = logging.getLogger(__name__)


DEFAULT_PURITY_THRESHOLD: Final[np.double] = np.sqrt(0.5, dtype=np.double)


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


def _run_and_evaluate(
    config: dict[str, Any],
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> tuple[dict[str, Any], list[str], list[NDArray[np.double]]]:
    """Run 1D IBU per variable and evaluate on test set."""
    splits: DatasetSplits = _load_splits(config)

    # Collect all splits for building response matrix (same as OmniFold)
    zs: list[NDArray[np.double]] = []
    xs: list[NDArray[np.double]] = []
    ys: list[NDArray[np.ubyte]] = []
    for split in [splits.train, splits.val, splits.test]:
        z_split, x_split, y_split = split.as_arrays()
        zs.append(z_split)
        xs.append(x_split)
        ys.append(y_split)
    z_all: NDArray[np.double] = np.concatenate(zs, axis=0)
    x_all: NDArray[np.double] = np.concatenate(xs, axis=0)
    y_all: NDArray[np.ubyte] = np.concatenate(ys, axis=0)

    z_gen_all: NDArray[np.double] = z_all[y_all == 0]
    x_sim_all: NDArray[np.double] = x_all[y_all == 0]
    x_data_all: NDArray[np.double] = x_all[y_all == 1]

    # Test split for evaluation
    z_test: NDArray[np.double]
    x_test: NDArray[np.double]
    y_test: NDArray[np.double]
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t: NDArray[np.double] = z_test[y_test == 1]
    x_data_t: NDArray[np.double] = x_test[y_test == 1]
    z_mc_t: NDArray[np.double] = z_test[y_test == 0]
    x_mc_t: NDArray[np.double] = x_test[y_test == 0]

    dim: int = z_all.shape[1]
    dataset: str = config.get("dataset", "gaussian")
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(dim)]

    metrics: dict = {}
    per_var_weights: list[NDArray[np.double]] = []

    for d in range(dim):
        # Purity-based binning from all MC
        bins: NDArray[np.double] = _purity_bins(
            z_gen_all[:, d],
            x_sim_all[:, d],
            purity_threshold,
        )
        n_bins: int = bins.shape[0] - 1

        if n_bins < 2:
            logger.warning("%s: only %d bin(s), skipping", var_names[d], n_bins)
            per_var_weights.append(np.ones(z_mc_t.shape[0], dtype=np.double))
            continue

        # Response matrix from all MC
        gen_binned: NDArray[np.long] = (
            np.clip(np.digitize(z_gen_all[:, d], bins), 1, n_bins) - 1
        )
        sim_binned: NDArray[np.long] = (
            np.clip(np.digitize(x_sim_all[:, d], bins), 1, n_bins) - 1
        )
        response: NDArray[np.double] = _build_response(gen_binned, sim_binned, n_bins)

        # Prior (MC gen) and data reco histogram
        # np.histogram returns integer counts; _ibu divides and accumulates into
        # these, so promote once here rather than relying on operand coercion.
        prior: NDArray[np.double] = np.histogram(z_gen_all[:, d], bins=bins)[0].astype(
            np.double
        )
        data_hist: NDArray[np.double] = np.histogram(x_data_all[:, d], bins=bins)[
            0
        ].astype(np.double)

        # IBU
        unfolded: NDArray[np.double] = _ibu(prior, data_hist, response, n_iterations)
        logger.info("%s: %d bins, %d iterations", var_names[d], n_bins, n_iterations)

        # Convert unfolded histogram to per-event weights for test MC.
        # Weight per bin = unfolded / prior; test MC events in that
        # gen-level bin receive the corresponding weight.
        bin_weights: NDArray[np.double] = unfolded / (prior + EPS)
        mc_test_binned: NDArray[np.long] = (
            np.clip(
                np.digitize(z_mc_t[:, d], bins),
                1,
                n_bins,
            )
            - 1
        )
        w: NDArray[np.double] = bin_weights[mc_test_binned]
        w: NDArray[np.double] = w / w.mean()
        per_var_weights.append(w)

        # Metrics per variable (IBU is 1D, so weights differ per variable)
        wd_before: list[float]
        wd_after: list[float]
        js_before: list[float]
        js_after: list[float]
        td_before: list[float]
        td_after: list[float]
        level: str
        ref: NDArray[np.double]
        comp: NDArray[np.double]
        key: str

        for level, ref, comp in [
            ("detector", x_data_t[:, d : d + 1], x_mc_t[:, d : d + 1]),
            ("particle", z_data_t[:, d : d + 1], z_mc_t[:, d : d + 1]),
        ]:
            wd_before = _wd_per_dim(ref, comp)
            wd_after = _wd_per_dim(ref, comp, weights=w)
            js_before = _js_per_dim(ref, comp)
            js_after = _js_per_dim(ref, comp, weights=w)
            td_before = _triangular_per_dim(ref, comp)
            td_after = _triangular_per_dim(ref, comp, weights=w)

            key = f"{level}_{var_names[d]}"
            metrics[key] = {
                "wasserstein_before": wd_before[0],
                "wasserstein_after": wd_after[0],
                "wasserstein_improvement_pct": _improvement(wd_before[0], wd_after[0]),
                "jensenshannon_before": js_before[0],
                "jensenshannon_after": js_after[0],
                "jensenshannon_improvement_pct": _improvement(
                    js_before[0], js_after[0]
                ),
                "triangular_before": td_before[0],
                "triangular_after": td_after[0],
                "triangular_improvement_pct": _improvement(td_before[0], td_after[0]),
            }

    return metrics, var_names, per_var_weights


def evaluate_single(
    run_dir: str | Path,
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> dict[str, Any]:
    """Run IBU on a single RAN run's dataset and save comparison metrics."""
    run_dir = Path(run_dir)
    out_path: Path = run_dir / "metrics_ibu.json"

    if out_path.exists() and not force:
        logger.info("%s: metrics_ibu.json exists, skipping (use --force)", run_dir.name)
        return json.loads(out_path.read_text())

    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    logger.info(
        "%s: running IBU (niter=%d, purity=%.4f)...",
        run_dir.name,
        n_iterations,
        purity_threshold,
    )
    metrics: dict[str, Any]
    var_names: list[str]
    per_var_weights: list[npt.NDArray[np.double]]

    metrics, var_names, per_var_weights = _run_and_evaluate(
        config,
        n_iterations=n_iterations,
        purity_threshold=purity_threshold,
    )

    json.dump(metrics, out_path.open("w"), indent=2)
    weights_path: Path = run_dir / "ibu_weights.npz"
    np.savez(
        weights_path,
        # savez is `savez(file, *args, allow_pickle=True, **kwds)`. Our keys are
        # built by f-string, so their type is plain `str` -- one of them *could*
        # be "allow_pickle", which is declared bool. The complaint is therefore
        # sound rather than a stub bug (ty reports it too); it just cannot happen
        # here. A literal-key dict checks clean, but ours cannot be one.
        **{f"weights_{i}": w for i, w in enumerate(per_var_weights)},  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    )
    logger.info(
        "%s: saved IBU metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [IBU]", metrics, var_names)
    return metrics


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
