"""IBU (Iterative Bayesian Unfolding) baseline for RAN comparison.

1D per-variable unfolding with purity-based automatic binning.
Builds the response matrix from MC, unfolds data, and converts
the result to per-event weights for evaluation with the same
metrics as RAN and OmniFold.

Usage:
    uv run -m ran baseline ibu --run-dir runs/2026-...
    uv run -m ran baseline ibu --run-dir runs  # all runs
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

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
    from ran.data import DatasetSplits

logger = logging.getLogger(__name__)


DEFAULT_PURITY_THRESHOLD = np.sqrt(0.5, dtype=np.double)


def _next_pure_edge(
    gen: npt.NDArray[np.double],
    reco: npt.NDArray[np.double],
    lo: np.double,
    purity_threshold: np.double,
) -> np.double | None:
    """Lowest edge above `lo` whose bin clears the purity threshold, else None.

    Scans 100 candidate edges between `lo` and the top of the gen range,
    returning the first that qualifies. None means no bin starting at `lo` is
    ever pure enough, which is what stops the caller's loop.
    """
    for binhigh in np.linspace(lo + 0.01, gen.max(), 100, dtype=np.double):
        in_truth = (gen >= lo) & (gen < binhigh)
        n_truth = np.sum(in_truth)
        if n_truth == 0:
            continue
        purity = np.sum(in_truth & (reco >= lo) & (reco < binhigh)) / n_truth
        if purity > purity_threshold:
            return binhigh
    return None


def _purity_bins(
    gen: npt.NDArray[np.double],
    reco: npt.NDArray[np.double],
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
    max_bins: int = 50,
) -> npt.NDArray:
    """Determine bin edges where purity exceeds threshold.

    Purity of a bin [lo, hi) = (events with truth AND reco in bin)
                              / (events with truth in bin).
    Bins are grown from the left edge until purity is met, then a new
    bin starts.
    """
    binvals: list[np.double] = [gen.min()]
    i = 0
    while binvals[-1] < gen.max() and i < len(binvals) and len(binvals) <= max_bins:
        edge = _next_pure_edge(gen, reco, binvals[i], purity_threshold)
        if edge is None:
            break
        binvals.append(edge)
        i += 1
    return np.array(binvals)


def _build_response(
    gen_bins: npt.NDArray[np.long],
    reco_bins: npt.NDArray[np.long],
    n_bins: int,
) -> npt.NDArray[np.double]:
    """Build row-normalized response matrix R[t,r] = P(reco=r | truth=t)."""
    response: npt.NDArray[np.double] = np.zeros((n_bins, n_bins), dtype=np.double)
    np.add.at(response, (gen_bins, reco_bins), 1)
    row_sums: npt.NDArray[np.double] = response.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    response /= row_sums
    return response


def _ibu(
    prior: npt.NDArray[np.double],
    data_hist: npt.NDArray[np.double],
    response: npt.NDArray[np.double],
    n_iterations: int,
) -> npt.NDArray:
    """Iterative Bayesian Unfolding.

    Args:
        prior: Initial truth estimate (MC gen histogram), shape (n_bins,).
        data_hist: Observed reco-level histogram, shape (n_bins,).
        response: R[t,r] = P(reco=r | truth=t), shape (n_bins, n_bins).
        n_iterations: Number of unfolding iterations.

    Returns:
        Unfolded truth histogram, shape (n_bins,).
    """
    posterior: npt.NDArray[np.double] = prior.copy()
    m: npt.NDArray[np.double]
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
) -> tuple[dict[str, Any], list[str], list[npt.NDArray[np.double]]]:
    """Run 1D IBU per variable and evaluate on test set."""
    splits: DatasetSplits = _load_splits(config)

    # Collect all splits for building response matrix (same as OmniFold)
    zs: list[npt.NDArray[np.double]] = []
    xs: list[npt.NDArray[np.double]] = []
    ys: list[npt.NDArray[np.ubyte]] = []
    for split in [splits.train, splits.val, splits.test]:
        z_split, x_split, y_split = split.as_arrays()
        zs.append(z_split)
        xs.append(x_split)
        ys.append(y_split)
    z_all: npt.NDArray[np.double] = np.concatenate(zs, axis=0)
    x_all: npt.NDArray[np.double] = np.concatenate(xs, axis=0)
    y_all: npt.NDArray[np.ubyte] = np.concatenate(ys, axis=0)

    z_gen_all: npt.NDArray[np.double] = z_all[y_all == 0]
    x_sim_all: npt.NDArray[np.double] = x_all[y_all == 0]
    x_data_all: npt.NDArray[np.double] = x_all[y_all == 1]

    # Test split for evaluation
    z_test: npt.NDArray[np.double]
    x_test: npt.NDArray[np.double]
    y_test: npt.NDArray[np.double]
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t: npt.NDArray[np.double] = z_test[y_test == 1]
    x_data_t: npt.NDArray[np.double] = x_test[y_test == 1]
    z_mc_t: npt.NDArray[np.double] = z_test[y_test == 0]
    x_mc_t: npt.NDArray[np.double] = x_test[y_test == 0]

    dim: int = z_all.shape[1]
    dataset: str = config.get("dataset", "gaussian")
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names: list[str] = [f"dim_{i}" for i in range(dim)]

    metrics: dict = {}
    per_var_weights: list[npt.NDArray[np.double]] = []

    for d in range(dim):
        # Purity-based binning from all MC
        bins: npt.NDArray[np.double] = _purity_bins(
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
        gen_binned: npt.NDArray[np.long] = (
            np.clip(np.digitize(z_gen_all[:, d], bins), 1, n_bins) - 1
        )
        sim_binned: npt.NDArray[np.long] = (
            np.clip(np.digitize(x_sim_all[:, d], bins), 1, n_bins) - 1
        )
        response: npt.NDArray[np.double] = _build_response(
            gen_binned, sim_binned, n_bins
        )

        # Prior (MC gen) and data reco histogram
        # np.histogram returns integer counts; _ibu divides and accumulates into
        # these, so promote once here rather than relying on operand coercion.
        prior: npt.NDArray[np.double] = np.histogram(z_gen_all[:, d], bins=bins)[
            0
        ].astype(np.double)
        data_hist: npt.NDArray[np.double] = np.histogram(x_data_all[:, d], bins=bins)[
            0
        ].astype(np.double)

        # IBU
        unfolded: npt.NDArray[np.double] = _ibu(
            prior, data_hist, response, n_iterations
        )
        logger.info("%s: %d bins, %d iterations", var_names[d], n_bins, n_iterations)

        # Convert unfolded histogram to per-event weights for test MC.
        # Weight per bin = unfolded / prior; test MC events in that
        # gen-level bin receive the corresponding weight.
        bin_weights: npt.NDArray[np.double] = unfolded / (prior + EPS)
        mc_test_binned: npt.NDArray[np.long] = (
            np.clip(
                np.digitize(z_mc_t[:, d], bins),
                1,
                n_bins,
            )
            - 1
        )
        w: npt.NDArray[np.double] = bin_weights[mc_test_binned]
        w: npt.NDArray[np.double] = w / w.mean()
        per_var_weights.append(w)

        # Metrics per variable (IBU is 1D, so weights differ per variable)
        wd_before: list[float]
        wd_after: list[float]
        js_before: list[float]
        js_after: list[float]
        td_before: list[float]
        td_after: list[float]
        level: str
        ref: npt.NDArray[np.double]
        comp: npt.NDArray[np.double]
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
