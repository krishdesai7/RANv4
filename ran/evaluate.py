"""Compute distance metrics on test sets for completed runs.

Computes per-dimension 1D Wasserstein distances and Jensen-Shannon
divergences, both before and after reweighting. Uses only memory-efficient
algorithms: sorted-CDF Wasserstein (O(n log n)) and histogram-based JS
divergence.

Usage:
    python -m ran.evaluate                          # all runs in runs/
    python -m ran.evaluate --run_dir=runs/2026-...  # single run
    python -m ran.evaluate --force                  # recompute existing
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import fire
import numpy as np
import numpy.typing as npt
import keras

from ran.data import RAN_Dataset, DatasetSplits
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def _load_splits(config: dict) -> DatasetSplits:
    """Reconstruct dataset splits from a run config."""
    dataset = config.get("dataset", "gaussian")
    n_samples = config["n_samples"]
    batch_size = config["batch_size"]
    dim = config["dim"]

    if dataset == "gaussian":
        gaussian_params = config["gaussian_params"]
        raw_params = {k: v for k, v in gaussian_params.items() if k != "dim"}
        return RAN_Dataset(batch_size=batch_size).generate_gaussian_dataset(
            params=raw_params, n_samples=n_samples,
        )
    elif dataset == "jets":
        from ran.data import load_jet_dataset
        splits, _, _ = load_jet_dataset(
            n_samples=n_samples, batch_size=batch_size,
            variables=tuple(config["variables"]),
        )
        return splits
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def _collect_test_data(test_ds):
    """Iterate test dataset in batches, return concatenated arrays."""
    zs, xs, ys = [], [], []
    for features, y in test_ds.as_numpy_iterator():
        zs.append(features["z"])
        xs.append(features["x"])
        ys.append(y)
    return (
        np.concatenate(zs, axis=0),
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0).reshape(-1),
    )


def _get_weights(g: keras.Model, z_gen: npt.NDArray, chunk_size: int = 10_000) -> npt.NDArray:
    """Compute normalized generator weights in chunks to limit peak memory."""
    n = len(z_gen)
    raw = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        raw[start:end] = g(z_gen[start:end]).numpy().flatten()
    return raw / raw.mean()


def _wd_per_dim(
    ref: npt.NDArray, comp: npt.NDArray, weights: npt.NDArray | None = None,
) -> list[float]:
    """1D Wasserstein distance per dimension using sorted-CDF fast path."""
    dim = ref.shape[1] if ref.ndim > 1 else 1
    result = []
    for i in range(dim):
        r = ref[:, i] if dim > 1 else ref.ravel()
        c = comp[:, i] if dim > 1 else comp.ravel()
        result.append(float(wasserstein_distance(r, c, v_weights=weights)))
    return result


def _js_per_dim(
    ref: npt.NDArray, comp: npt.NDArray,
    weights: npt.NDArray | None = None, n_bins: int = 100,
) -> list[float]:
    """Jensen-Shannon divergence per dimension via histogramming.

    Returns JS divergence (squared JS distance) per dimension.
    """
    dim = ref.shape[1] if ref.ndim > 1 else 1
    result = []
    for i in range(dim):
        r = ref[:, i] if dim > 1 else ref.ravel()
        c = comp[:, i] if dim > 1 else comp.ravel()

        lo = min(r.min(), c.min())
        hi = max(r.max(), c.max())
        bins = np.linspace(lo, hi, n_bins + 1)

        h_ref, _ = np.histogram(r, bins=bins)
        if weights is not None:
            h_comp, _ = np.histogram(c, bins=bins, weights=weights)
        else:
            h_comp, _ = np.histogram(c, bins=bins)

        # Normalize to probability distributions
        h_ref = h_ref.astype(np.float64)
        h_comp = h_comp.astype(np.float64)
        s_ref = h_ref.sum()
        s_comp = h_comp.sum()
        if s_ref > 0:
            h_ref /= s_ref
        if s_comp > 0:
            h_comp /= s_comp

        # JS divergence = (JS distance)^2
        result.append(float(jensenshannon(h_ref, h_comp) ** 2))
    return result


def _triangular_per_dim(
    ref: npt.NDArray, comp: npt.NDArray,
    weights: npt.NDArray | None = None, n_bins: int = 100,
) -> list[float]:
    """Triangular discriminator (Vincze-LeCam divergence) per dimension.

    Δ(p,q) = Σ (p_i - q_i)² / (p_i + q_i)  ×  1e3

    where p_i, q_i are histogram probability masses. The bin-width factor
    cancels analytically, so this works directly on normalized histograms.
    """
    dim = ref.shape[1] if ref.ndim > 1 else 1
    result = []
    for i in range(dim):
        r = ref[:, i] if dim > 1 else ref.ravel()
        c = comp[:, i] if dim > 1 else comp.ravel()

        lo = min(r.min(), c.min())
        hi = max(r.max(), c.max())
        bins = np.linspace(lo, hi, n_bins + 1)

        h_ref, _ = np.histogram(r, bins=bins)
        if weights is not None:
            h_comp, _ = np.histogram(c, bins=bins, weights=weights)
        else:
            h_comp, _ = np.histogram(c, bins=bins)

        # Normalize to probability mass
        h_ref = h_ref.astype(np.float64)
        h_comp = h_comp.astype(np.float64)
        s_ref = h_ref.sum()
        s_comp = h_comp.sum()
        if s_ref > 0:
            h_ref /= s_ref
        if s_comp > 0:
            h_comp /= s_comp

        denom = h_ref + h_comp
        mask = denom > 0
        diff = h_ref - h_comp
        result.append(float(np.sum(diff[mask] ** 2 / denom[mask]) * 1e3))
    return result


def _improvement(before: float, after: float) -> float:
    return (1 - after / before) * 100 if before > 0 else 0.0


def evaluate_run(run_dir: str | Path, force: bool = False) -> dict:
    """Evaluate a single run directory."""
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics.json"

    if out_path.exists() and not force:
        print(f"  {run_dir.name}: metrics.json exists, skipping (use --force)")
        return json.loads(out_path.read_text())

    config = json.loads((run_dir / "config.json").read_text())
    print(f"  {run_dir.name}: loading model and data...")
    g = keras.saving.load_model(run_dir / "generator.keras")

    splits = _load_splits(config)
    z, x, y = _collect_test_data(splits.test)

    z_data, z_mc = z[y == 1], z[y == 0]
    x_data, x_mc = x[y == 1], x[y == 0]
    w = _get_weights(g, z_mc)

    # Variable names for labeling
    dataset = config.get("dataset", "gaussian")
    dim = config["dim"]
    if dataset == "jets":
        var_names = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(dim)]

    metrics: dict = {}

    for level, data, mc in [("detector", x_data, x_mc), ("particle", z_data, z_mc)]:
        wd_before = _wd_per_dim(data, mc)
        wd_after = _wd_per_dim(data, mc, weights=w)
        js_before = _js_per_dim(data, mc)
        js_after = _js_per_dim(data, mc, weights=w)
        td_before = _triangular_per_dim(data, mc)
        td_after = _triangular_per_dim(data, mc, weights=w)

        for i, var in enumerate(var_names):
            key = f"{level}_{var}"
            metrics[key] = {
                "wasserstein_before": wd_before[i],
                "wasserstein_after": wd_after[i],
                "wasserstein_improvement_pct": _improvement(wd_before[i], wd_after[i]),
                "jensenshannon_before": js_before[i],
                "jensenshannon_after": js_after[i],
                "jensenshannon_improvement_pct": _improvement(js_before[i], js_after[i]),
                "triangular_before": td_before[i],
                "triangular_after": td_after[i],
                "triangular_improvement_pct": _improvement(td_before[i], td_after[i]),
            }

    json.dump(metrics, out_path.open("w"), indent=2)
    _print_metrics(run_dir.name, metrics, var_names)
    return metrics


def _print_metrics(run_name: str, metrics: dict, var_names: list[str]) -> None:
    print(f"\n{'='*72}")
    print(f"  Run: {run_name}")
    print(f"{'='*72}")

    for level in ("detector", "particle"):
        print(f"\n  {level.upper()} LEVEL")
        print(f"  {'Var':<10} {'Metric':<14} {'Before':>10} {'After':>10} {'Improv.':>10}")
        print(f"  {'-'*56}")
        for var in var_names:
            key = f"{level}_{var}"
            if key not in metrics:
                continue
            m = metrics[key]
            print(
                f"  {var:<10} {'Wasserstein':<14}"
                f" {m['wasserstein_before']:>10.4f}"
                f" {m['wasserstein_after']:>10.4f}"
                f" {m['wasserstein_improvement_pct']:>+9.1f}%"
            )
            print(
                f"  {'':<10} {'JS div':<14}"
                f" {m['jensenshannon_before']:>10.6f}"
                f" {m['jensenshannon_after']:>10.6f}"
                f" {m['jensenshannon_improvement_pct']:>+9.1f}%"
            )
            print(
                f"  {'':<10} {chr(916)+' (x1e3)':<14}"
                f" {m['triangular_before']:>10.4f}"
                f" {m['triangular_after']:>10.4f}"
                f" {m['triangular_improvement_pct']:>+9.1f}%"
            )


def main(run_dir: str = "runs", force: bool = False):
    """Compute distance metrics for completed runs.

    Args:
        run_dir: Path to a single run or a directory containing multiple runs.
        force: Recompute even if metrics.json already exists.
    """
    run_dir = Path(run_dir)

    if (run_dir / "config.json").exists():
        evaluate_run(run_dir, force=force)
    else:
        run_dirs = sorted(
            d for d in run_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
        print(f"Found {len(run_dirs)} runs to evaluate")
        for d in run_dirs:
            try:
                evaluate_run(d, force=force)
            except Exception as e:
                print(f"  {d.name}: FAILED — {e}")


if __name__ == "__main__":
    fire.Fire(main)
