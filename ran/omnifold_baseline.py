"""Run OmniFold on the same dataset as a RAN run for comparison.

Usage:
    python -m ran.omnifold_baseline --run_dir=runs/2026-...
    python -m ran.omnifold_baseline --run_dir=runs  # all runs
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import fire
import numpy as np
import keras
from omnifold import MultiFold, DataLoader, MLP
from omnifold.net import weighted_binary_crossentropy

# OmniFold's custom loss isn't registered with Keras serialization,
# which breaks clone_model(). Register it here.
keras.saving.get_custom_objects()["weighted_binary_crossentropy"] = weighted_binary_crossentropy

from ran.evaluate import (
    _load_splits, _collect_test_data,
    _wd_per_dim, _js_per_dim, _triangular_per_dim,
    _improvement, _print_metrics,
)


def _run_and_evaluate(config: dict, niter: int = 3, epochs: int = 50) -> tuple[dict, list[str]]:
    """Train OmniFold on a RAN dataset and evaluate on test set."""
    splits = _load_splits(config)

    # Collect all splits into flat arrays for OmniFold training
    zs, xs, ys = [], [], []
    for split in [splits.train, splits.val, splits.test]:
        for features, y_batch in split:
            zs.append(features["z"].numpy())
            xs.append(features["x"].numpy())
            ys.append(y_batch.numpy().reshape(-1))
    z_all = np.concatenate(zs, axis=0)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    mask_data = y_all == 1
    mask_mc = y_all == 0

    x_data = x_all[mask_data].astype(np.float32)
    x_mc = x_all[mask_mc].astype(np.float32)
    z_mc = z_all[mask_mc].astype(np.float32)

    dim = x_data.shape[1]

    data_dl = DataLoader(reco=x_data)
    mc_dl = DataLoader(reco=x_mc, gen=z_mc)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(dim), MLP(dim),
        data_dl, mc_dl,
        niter=niter,
        epochs=epochs,
        batch_size=512,
        verbose=False,
    )
    unfold.Unfold()

    # Evaluate on test split only
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t = z_test[y_test == 1]
    x_data_t = x_test[y_test == 1]
    z_mc_t = z_test[y_test == 0]
    x_mc_t = x_test[y_test == 0]

    # Get OmniFold weights for test MC via the trained gen-level model
    w = unfold.reweight(z_mc_t.astype(np.float32), unfold.model2).astype(np.float64)
    w = w * len(w) / w.sum()  # normalize to preserve total count

    dataset = config.get("dataset", "gaussian")
    if dataset == "jets":
        var_names = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(config["dim"])]

    metrics: dict = {}
    for level, ref, comp in [("detector", x_data_t, x_mc_t), ("particle", z_data_t, z_mc_t)]:
        wd_before = _wd_per_dim(ref, comp)
        wd_after = _wd_per_dim(ref, comp, weights=w)
        js_before = _js_per_dim(ref, comp)
        js_after = _js_per_dim(ref, comp, weights=w)
        td_before = _triangular_per_dim(ref, comp)
        td_after = _triangular_per_dim(ref, comp, weights=w)

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

    return metrics, var_names


def evaluate_single(run_dir: str | Path, force: bool = False,
                    niter: int = 3, epochs: int = 50) -> dict:
    """Run OmniFold on a single RAN run's dataset and save comparison metrics."""
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics_omnifold.json"

    if out_path.exists() and not force:
        print(f"  {run_dir.name}: metrics_omnifold.json exists, skipping (use --force)")
        return json.loads(out_path.read_text())

    config = json.loads((run_dir / "config.json").read_text())
    print(f"  {run_dir.name}: running OmniFold (niter={niter}, epochs={epochs})...")

    metrics, var_names = _run_and_evaluate(config, niter=niter, epochs=epochs)

    json.dump(metrics, out_path.open("w"), indent=2)
    _print_metrics(f"{run_dir.name} [OmniFold]", metrics, var_names)
    return metrics


def main(run_dir: str = "runs", force: bool = False,
         niter: int = 3, epochs: int = 50):
    """Run OmniFold baseline on completed RAN runs.

    Args:
        run_dir: Path to a single run or directory of runs.
        force: Recompute even if metrics_omnifold.json exists.
        niter: Number of OmniFold iterations.
        epochs: Max epochs per OmniFold iteration.
    """
    run_dir = Path(run_dir)

    if (run_dir / "config.json").exists():
        evaluate_single(run_dir, force=force, niter=niter, epochs=epochs)
    else:
        run_dirs = sorted(
            d for d in run_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
        print(f"Found {len(run_dirs)} runs to evaluate with OmniFold")
        for d in run_dirs:
            try:
                evaluate_single(d, force=force, niter=niter, epochs=epochs)
            except Exception as e:
                print(f"  {d.name}: FAILED — {e}")


if __name__ == "__main__":
    fire.Fire(main)
