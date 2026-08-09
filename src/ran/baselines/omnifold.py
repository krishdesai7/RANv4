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
    *,
    out_dir: Path,
) -> NDArray[np.single]:
    """Unfold `x_sim`/`z_gen` toward `x_data` with OmniFold, returning per-event
    weights normalized to mean 1.

    `out_dir` is where the `omnifold` library scatters its own bookkeeping:
    `MultiFold` opens ``log_<name>.txt`` in `log_folder` and dumps a checkpoint
    per iteration/step into `weights_folder`, both defaulting to the process
    cwd. It is keyword-only and has no default on purpose -- every caller must
    say where those land, or concurrent callers sharing a cwd silently
    overwrite each other's files (the sweep runs up to 24 points at once from
    one working directory). Give each concurrent call its own directory.
    """
    x_data = _as2d(x_data)
    x_sim = _as2d(x_sim)
    z_gen = _as2d(z_gen)
    z_target = z_gen if z_target is None else _as2d(z_target)
    dim = x_data.shape[1]

    data_dl = DataLoader(reco=x_data)
    mc_dl = DataLoader(reco=x_sim, gen=z_gen)

    # MultiFold opens its log file at construction and does not create the
    # folder first, so it has to exist by now. It does create weights_folder.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(dim),
        MLP(dim),
        data_dl,
        mc_dl,
        log_folder=str(out_dir),
        # Not "omnifold_weights": that is our own result artifact
        # (omnifold_weights.npz). These are the library's per-iteration
        # checkpoints, which nothing reads back -- `MultiFold.LoadStart` only
        # reloads them when resuming from `start > 0`, which we never do.
        weights_folder=str(out_dir / "omnifold_checkpoints"),
        niter=niter,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
    )
    unfold.Unfold()

    w: NDArray[np.single] = (
        unfold.reweight(z_target, unfold.model2).astype(np.single).ravel()
    )
    return np.divide(w, w.mean(), dtype=np.single)


def _run_and_evaluate(
    config: RunConfig, niter: int = 3, epochs: int = 50, out_dir: Path = Path()
) -> tuple[dict[str, MetricRecord], list[str], NDArray[np.single]]:
    """Train OmniFold on a RAN dataset and evaluate on test set."""
    full, test = load_populations(config)
    test_truth = test.require_truth()

    # OmniFold trains under TensorFlow, so cast the shared float64 populations
    # to float32 here rather than making every baseline pay for it.
    w: NDArray[np.single] = omnifold_unfold(
        _as2d(full.data),
        _as2d(full.mc.x),
        _as2d(full.mc.z),
        z_target=_as2d(test.mc.z),
        niter=niter,
        epochs=epochs,
        out_dir=out_dir,
    )

    # One joint weight vector covers every dimension, unlike IBU's per-variable
    # weights -- OmniFold unfolds all observables together.
    metrics: dict[str, MetricRecord] = {}
    for dimension, variable_name in enumerate(config.variable_names):
        metrics[f"detector_{variable_name}"] = evaluate_dimension(
            test.data[:, dimension],
            test.mc.x[:, dimension],
            w,
        )
        metrics[f"particle_{variable_name}"] = evaluate_dimension(
            test_truth[:, dimension],
            test.mc.z[:, dimension],
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

    config: RunConfig = parse_run_config(
        json.loads((run_dir / "config.json").read_text())
    )
    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...", run_dir.name, niter, epochs
    )

    metrics, var_names, w = _run_and_evaluate(
        config, niter=niter, epochs=epochs, out_dir=run_dir
    )

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
