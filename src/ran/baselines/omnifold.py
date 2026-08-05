"""Run OmniFold on the same dataset as a RAN run for comparison.

Usage:
    uv run -m ran baseline omnifold --run-dir runs/2026-...
    uv run -m ran baseline omnifold --run-dir runs  # all runs

RAN itself runs on the JAX backend, but the third-party `omnifold` package does
not: its `weighted_binary_crossentropy` calls raw `tf.gather` on the label
tensor, which raises `TracerArrayConversionError` the moment JAX traces it. So
this module pins the backend back to TensorFlow.

A process gets one Keras backend, set at first `keras` import, so invoke the
OmniFold baseline in its own process with `uv run -m ran baseline omnifold`;
never import it from a module that has already touched JAX. The cubic sweep
keeps the two sides in separate subcommands for exactly this reason.
"""

import os

from ran.evaluate import (
    _collect_test_data,
    _improvement,
    _js_per_dim,
    _load_splits,
    _triangular_per_dim,
    _wd_per_dim,
    render_metrics,
)

# Must precede every keras import, including the transitive one via `ran`
# (whose __init__ only *defaults* the backend to jax, so this hard set wins).
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import logging
from pathlib import Path

import keras
import numpy as np
import numpy.typing as npt
from omnifold import MLP, DataLoader, MultiFold
from omnifold.net import weighted_binary_crossentropy

# OmniFold's custom loss isn't registered with Keras serialization,
# which breaks clone_model(). Register it here.
keras.saving.get_custom_objects()["weighted_binary_crossentropy"] = (
    weighted_binary_crossentropy
)


logger = logging.getLogger(__name__)


def omnifold_unfold(
    x_data: npt.NDArray,
    x_sim: npt.NDArray,
    z_gen: npt.NDArray,
    z_target: npt.NDArray | None = None,
    niter: int = 3,
    epochs: int = 50,
    batch_size: int = 512,
) -> npt.NDArray[np.float64]:
    """Train OmniFold on in-memory arrays; return mean-normalized gen weights.

    Trains on (data reco = x_data, MC reco = x_sim, MC gen = z_gen), then
    reweights z_target (defaults to z_gen) through the gen-level model. Returns
    a 1D weight array, normalized so its mean is 1.
    """

    def _as2d(a: npt.NDArray) -> npt.NDArray:
        a = np.asarray(a, dtype=np.float32)
        return a[:, None] if a.ndim == 1 else a

    x_data = _as2d(x_data)
    x_sim = _as2d(x_sim)
    z_gen = _as2d(z_gen)
    z_target = z_gen if z_target is None else _as2d(z_target)
    dim = x_data.shape[1]

    data_dl = DataLoader(reco=x_data)
    mc_dl = DataLoader(reco=x_sim, gen=z_gen)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(dim),
        MLP(dim),
        data_dl,
        mc_dl,
        niter=niter,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
    )
    unfold.Unfold()

    w = unfold.reweight(z_target, unfold.model2).astype(np.float64).ravel()
    return w / w.mean()


def _run_and_evaluate(
    config: dict, niter: int = 3, epochs: int = 50
) -> tuple[dict, list[str], npt.NDArray[np.float64]]:
    """Train OmniFold on a RAN dataset and evaluate on test set."""
    splits = _load_splits(config)

    # Collect all splits into flat arrays for OmniFold training
    zs, xs, ys = [], [], []
    for split in [splits.train, splits.val, splits.test]:
        z_split, x_split, y_split = split.as_arrays()
        zs.append(z_split)
        xs.append(x_split)
        ys.append(y_split)
    z_all = np.concatenate(zs, axis=0)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    mask_data = y_all == 1
    mask_mc = y_all == 0

    x_data = x_all[mask_data].astype(np.float32)
    x_mc = x_all[mask_mc].astype(np.float32)
    z_mc = z_all[mask_mc].astype(np.float32)

    # Evaluate on test split only
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t = z_test[y_test == 1]
    x_data_t = x_test[y_test == 1]
    z_mc_t = z_test[y_test == 0]
    x_mc_t = x_test[y_test == 0]

    w = omnifold_unfold(
        x_data,
        x_mc,
        z_mc,
        z_target=z_mc_t,
        niter=niter,
        epochs=epochs,
    )

    dataset = config.get("dataset", "gaussian")
    if dataset == "jets":
        var_names = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(config["dim"])]

    metrics: dict = {}
    for level, ref, comp in [
        ("detector", x_data_t, x_mc_t),
        ("particle", z_data_t, z_mc_t),
    ]:
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
                "jensenshannon_improvement_pct": _improvement(
                    js_before[i], js_after[i]
                ),
                "triangular_before": td_before[i],
                "triangular_after": td_after[i],
                "triangular_improvement_pct": _improvement(td_before[i], td_after[i]),
            }

    return metrics, var_names, w


def evaluate_single(
    run_dir: str | Path, force: bool = False, niter: int = 3, epochs: int = 50
) -> dict:
    """Run OmniFold on a single RAN run's dataset and save comparison metrics."""
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics_omnifold.json"

    if out_path.exists() and not force:
        logger.info(
            "%s: metrics_omnifold.json exists, skipping (use --force)", run_dir.name
        )
        return json.loads(out_path.read_text())

    config = json.loads((run_dir / "config.json").read_text())
    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...", run_dir.name, niter, epochs
    )

    metrics, var_names, w = _run_and_evaluate(config, niter=niter, epochs=epochs)

    json.dump(metrics, out_path.open("w"), indent=2)
    weights_path = run_dir / "omnifold_weights.npz"
    np.savez(weights_path, weights=w)
    logger.info(
        "%s: saved OmniFold metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [OmniFold]", metrics, var_names)
    return metrics


def evaluate_runs(
    run_dir: str | Path = "runs",
    force: bool = False,
    niter: int = 3,
    epochs: int = 50,
) -> None:
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
            d for d in run_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
        )
        logger.info("Found %d runs to evaluate with OmniFold", len(run_dirs))
        for d in run_dirs:
            try:
                evaluate_single(d, force=force, niter=niter, epochs=epochs)
            except Exception:
                logger.warning("%s: failed", d.name, exc_info=True)
