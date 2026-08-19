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
os.environ.setdefault(key="TF_CPP_MIN_LOG_LEVEL", value="2")

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

    from ..rantypes import MetricRecord, Populations, RunConfig

# OmniFold's custom loss isn't registered with Keras serialization,
# which breaks clone_model(). Register it here.
keras.saving.get_custom_objects()["weighted_binary_crossentropy"] = (
    weighted_binary_crossentropy
)


logger: Logger = logging.getLogger(__name__)


def _as2d(a: ArrayLike, /) -> NDArray[np.single]:
    a2d: NDArray[np.single] = np.asarray(a, dtype=np.single)
    return a2d[..., np.newaxis] if a2d.ndim == 1 else a2d


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
    x_d: NDArray[np.single] = _as2d(x_data)
    x_s: NDArray[np.single] = _as2d(x_sim)
    z_g: NDArray[np.single] = _as2d(z_gen)
    z_t: NDArray[np.single] = z_g if z_target is None else _as2d(z_target)
    dim: int = x_d.shape[1]

    data_dl = DataLoader(reco=x_d)
    mc_dl = DataLoader(reco=x_s, gen=z_g)

    # MultiFold opens its log file at construction and does not create the
    # folder first, so it has to exist by now. It does create weights_folder.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unfold = MultiFold(
        name="omnifold_baseline",
        model_reco=MLP(nvars=dim),
        model_gen=MLP(nvars=dim),
        data=data_dl,
        mc=mc_dl,
        log_folder=str(out_dir),
        # Not "omnifold_weights": that is RAN's result artifact (omnifold_weights.npz)
        # These are OmniFold's per-iteration checkpoints, which nothing reads back.
        weights_folder=str(out_dir / "omnifold_checkpoints"),
        niter=niter,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
    )
    unfold.Unfold()

    w: NDArray[np.single] = (
        unfold.reweight(events=z_t, model=unfold.model2).astype(np.single).ravel()
    )
    return np.divide(w, w.mean(), dtype=np.single)


def _run_and_evaluate(
    config: RunConfig, niter: int = 3, epochs: int = 50, out_dir: Path = Path(), /
) -> tuple[dict[str, MetricRecord], list[str], NDArray[np.single]]:
    """Train OmniFold on a dataset and evaluate on test set."""
    full: Populations[np.single]
    test: Populations[np.single]
    full, test = load_populations(config).astype(np.single)
    test_truth: NDArray[np.single] = test.require_truth()

    w: NDArray[np.single] = omnifold_unfold(
        x_data=full.data,
        x_sim=full.mc.x,
        z_gen=full.mc.z,
        z_target=test_truth,
        niter=niter,
        epochs=epochs,
        out_dir=out_dir,
    )

    # One joint weight vector covers every dimension, unlike IBU's per-variable
    # weights -- OmniFold unfolds all observables together.
    metrics: dict[str, MetricRecord] = {}
    for dimension, variable_name in enumerate(iterable=config.variable_names):
        metrics[f"detector_{variable_name}"] = evaluate_dimension(
            reference=test.data[:, dimension],
            comparison=test.mc.x[:, dimension],
            weights=w,
        )
        metrics[f"particle_{variable_name}"] = evaluate_dimension(
            reference=test_truth[:, dimension],
            comparison=test.mc.z[:, dimension],
            weights=w,
        )

    return metrics, list(config.variable_names), w


def evaluate_single(
    run_dir: Path, force: bool = False, niter: int = 3, epochs: int = 50, /
) -> dict:
    """Run OmniFold on a single run's dataset and save comparison metrics."""
    out_path: Path = run_dir / "metrics_omnifold.json"

    if out_path.exists() and not force:
        logger.info(
            "%s: metrics_omnifold.json exists, skipping (use --force)", run_dir.name
        )
        return json.loads(out_path.read_text())

    config: RunConfig = parse_run_config(
        raw=json.loads(s=(run_dir / "config.json").read_text())
    )
    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...", run_dir.name, niter, epochs
    )

    metrics, var_names, w = _run_and_evaluate(config, niter, epochs, run_dir)

    json.dump(obj=metrics, fp=out_path.open(mode="w"), indent=2)
    weights_path: Path = run_dir / "omnifold_weights.npz"
    np.savez(file=weights_path, weights=w)
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
    apply_to_runs(
        run_dir,
        evaluate_one=lambda d: evaluate_single(d, force, niter, epochs),
        description="evaluate with OmniFold",
        log=logger,
    )
