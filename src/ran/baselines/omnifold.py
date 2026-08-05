from __future__ import annotations

import os

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

# Must precede every keras import, including the transitive one via `ran`
# (whose __init__ only *defaults* the backend to jax, so this hard set wins).
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import keras
import numpy as np
from omnifold import MLP, DataLoader, MultiFold
from omnifold.net import weighted_binary_crossentropy

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    from ran.data import DatasetSplits

# OmniFold's custom loss isn't registered with Keras serialization,
# which breaks clone_model(). Register it here.
keras.saving.get_custom_objects()["weighted_binary_crossentropy"] = (
    weighted_binary_crossentropy
)


logger: Logger = logging.getLogger(__name__)


def _as2d(a: ArrayLike) -> NDArray[np.single]:
    a = np.asarray(a, dtype=np.single)
    return a[..., np.newaxis] if a.ndim == 1 else a


def omnifold_unfold(
    x_data: NDArray[np.single],
    x_sim: NDArray[np.single],
    z_gen: NDArray[np.single],
    z_target: NDArray[np.single] | None = None,
    niter: int = 3,
    epochs: int = 50,
    batch_size: int = 512,
) -> NDArray[np.single]:

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

    w: NDArray[np.single] = (
        unfold.reweight(z_target, unfold.model2).astype(np.single).ravel()
    )
    return w / w.mean()


def _run_and_evaluate(
    config: dict[str, Any], niter: int = 3, epochs: int = 50
) -> tuple[dict, list[str], NDArray[np.single]]:
    """Train OmniFold on a RAN dataset and evaluate on test set."""
    splits: DatasetSplits = _load_splits(config)

    # Collect all splits into flat arrays for OmniFold training
    zs: list[NDArray[np.single]] = []
    xs: list[NDArray[np.single]] = []
    ys: list[NDArray[np.ubyte]] = []
    for split in [splits.train, splits.val, splits.test]:
        z_split, x_split, y_split = split.as_arrays()
        zs.append(_as2d(z_split))
        xs.append(_as2d(x_split))
        ys.append(y_split)
    z_all: NDArray[np.single] = np.concatenate(zs, axis=0)
    x_all: NDArray[np.single] = np.concatenate(xs, axis=0)
    y_all: NDArray[np.ubyte] = np.concatenate(ys, axis=0)

    mask_data: NDArray[np.bool] = y_all == 1

    x_data: NDArray[np.single] = x_all[mask_data]
    x_mc: NDArray[np.single] = x_all[~mask_data]
    z_mc: NDArray[np.single] = z_all[~mask_data]

    # Evaluate on test split only
    z_test, x_test, y_test = _collect_test_data(splits.test)
    mask_data_t: NDArray[np.bool] = y_test == 1
    z_data_t: NDArray[np.single] = z_test[mask_data_t]
    x_data_t: NDArray[np.single] = x_test[mask_data_t]
    z_mc_t: NDArray[np.single] = z_test[~mask_data_t]
    x_mc_t: NDArray[np.single] = x_test[~mask_data_t]

    w: NDArray[np.single] = omnifold_unfold(
        x_data,
        x_mc,
        z_mc,
        z_target=z_mc_t,
        niter=niter,
        epochs=epochs,
    )

    dataset: str = config.get("dataset", "gaussian")
    if dataset == "jets":
        var_names: list[str] = config["variables"]
    else:
        var_names = [f"dim_{i}" for i in range(config["dim"])]

    metrics: dict = {}
    for level, ref, comp in [
        ("detector", x_data_t, x_mc_t),
        ("particle", z_data_t, z_mc_t),
    ]:
        wd_before: list[float] = _wd_per_dim(ref, comp)
        wd_after: list[float] = _wd_per_dim(ref, comp, weights=w)
        js_before: list[float] = _js_per_dim(ref, comp)
        js_after: list[float] = _js_per_dim(ref, comp, weights=w)
        td_before: list[float] = _triangular_per_dim(ref, comp)
        td_after: list[float] = _triangular_per_dim(ref, comp, weights=w)

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

    return metrics, var_names, w


def evaluate_single(
    run_dir: Path, force: bool = False, niter: int = 3, epochs: int = 50
) -> dict:
    """Run OmniFold on a single RAN run's dataset and save comparison metrics."""
    out_path: Path = run_dir / "metrics_omnifold.json"

    if out_path.exists() and not force:
        logger.info(
            "%s: metrics_omnifold.json exists, skipping (use --force)", run_dir.name
        )
        return json.loads(out_path.read_text())

    config: dict = json.loads((run_dir / "config.json").read_text())
    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...", run_dir.name, niter, epochs
    )

    metrics, var_names, w = _run_and_evaluate(config, niter=niter, epochs=epochs)

    json.dump(metrics, out_path.open("w"), indent=2)
    weights_path: Path = run_dir / "omnifold_weights.npz"
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
    run_dir: Path = Path("runs"),
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
    apply_to_runs(
        run_dir,
        lambda d: evaluate_single(d, force=force, niter=niter, epochs=epochs),
        "evaluate with OmniFold",
        logger,
    )
