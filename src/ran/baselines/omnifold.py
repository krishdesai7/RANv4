from __future__ import annotations

import os

from ..evaluate import apply_to_runs, render_metrics
from ._shared import (
    evaluate_dimension,
    load_populations,
    parse_run_config,
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

    from numpy.typing import ArrayLike, NDArray

    from ..rantypes import MetricRecord, RunConfig

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
    x_data: ArrayLike,
    x_sim: ArrayLike,
    z_gen: ArrayLike,
    z_target: ArrayLike | None = None,
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
    config: RunConfig, niter: int = 3, epochs: int = 50
) -> tuple[dict[str, MetricRecord], list[str], NDArray[np.single]]:
    """Train OmniFold on a RAN dataset and evaluate on test set."""
    data = load_populations(config)

    # OmniFold trains under TensorFlow, so cast the shared float64 populations
    # to float32 here rather than making every baseline pay for it.
    w: NDArray[np.single] = omnifold_unfold(
        _as2d(data.observed_reco),
        _as2d(data.response_sim),
        _as2d(data.response_gen),
        z_target=_as2d(data.test_mc_gen),
        niter=niter,
        epochs=epochs,
    )

    # One joint weight vector covers every dimension, unlike IBU's per-variable
    # weights -- OmniFold unfolds all observables together.
    metrics: dict[str, MetricRecord] = {}
    for dimension, variable_name in enumerate(config.variable_names):
        metrics[f"detector_{variable_name}"] = evaluate_dimension(
            data.test_data_reco[:, dimension],
            data.test_mc_reco[:, dimension],
            w,
        )
        metrics[f"particle_{variable_name}"] = evaluate_dimension(
            data.test_data_gen[:, dimension],
            data.test_mc_gen[:, dimension],
            w,
        )

    return metrics, list(config.variable_names), w


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

    config = parse_run_config(json.loads((run_dir / "config.json").read_text()))
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
