from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import keras
import numpy as np

from .baselines import parse_run_config
from .data import (
    RANDataset,
    gaussian_config_from_run_config,
    load_jet_dataset,
    parse_gaussian_config,
)
from .evaluate import evaluate_run
from .mmd import bandwidths, build_cache, mmd_curve, subsample_indices
from .plotting import (
    plot_levels,
    plot_losses,
    plot_selection,
)
from .rantypes import (
    JET_OBS,
    DatasetName,
    GaussianConfig,
    VarInfo,
)
from .timing import phase, report, write
from .train import MMD_SUBSAMPLE, _weights_per_epoch, save_params, train

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any

    from .rantypes import DatasetSplits, EventArray, Populations, RANModel, RunConfig
    from .train import EpochParams, TrainResult

logger: Logger = logging.getLogger(__name__)


def _prepare_gaussian(
    config: Path | None,
    saved_config: GaussianConfig | None,
    batch_size: int,
    n_samples: int,
    data_seed: int,
) -> tuple[DatasetSplits, int, GaussianConfig]:
    builder: RANDataset = RANDataset(batch_size=batch_size, seed=data_seed)
    if saved_config is not None:
        gaussian_params: GaussianConfig = saved_config
        # Reload: use stored params from config.json
        splits: DatasetSplits = builder.generate_gaussian_dataset(
            params=saved_config, n_samples=n_samples
        )
    else:
        # Fresh run: parse YAML config
        if config is None:
            raise ValueError("Gaussian mode requires --config path/to/config.yaml")
        gaussian_params = parse_gaussian_config(config)
        splits = builder.generate_gaussian_dataset(
            config_path=config, n_samples=n_samples
        )
    return splits, gaussian_params.dim, gaussian_params


def _prepare_jets(
    n_samples: int,
    batch_size: int,
    variables: tuple[str, ...],
    data_seed: int,
) -> tuple[DatasetSplits, int, list[VarInfo]]:
    """Build jet splits plus the per-variable metadata the plots need."""
    std_params: dict[str, tuple[np.single, np.single]]
    splits, dim, std_params = load_jet_dataset(
        n_samples=n_samples,
        batch_size=batch_size,
        variables=variables,
        seed=data_seed,
    )
    var_info: list[VarInfo] = [
        VarInfo(
            xlim=JET_OBS[v].xlim,
            xlabel=JET_OBS[v].xlabel,
            symbol=JET_OBS[v].symbol,
            mu=float(std_params[v][0]),
            sigma=float(std_params[v][1]),
        )
        for v in variables
    ]
    return splits, dim, var_info


def _new_run_dir(explicit: Path | None) -> Path:
    """Choose the directory this run's artifacts land in, and claim it.

    A sweep names each run so its arm can be told from the others afterwards.
    An explicit directory is claimed by holding a `config.json`, not by
    existing: the launcher redirects each run's log into the directory before
    training starts, so it is routinely already there and empty.

    The default is a UTC timestamp at second resolution, which is not enough on
    its own. A packed sweep starts runs of identical shape together and they
    finish inside the same second; the `exist_ok=True` this replaces let the
    later run overwrite the earlier one with no error, no warning, and no way
    to tell afterwards which arm had gone missing.
    """
    if explicit is not None:
        if (explicit / "config.json").exists():
            raise FileExistsError(f"{explicit} already holds a finished run")
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit

    stamp: str = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H%M%SZ")
    candidate: Path = Path("runs") / stamp
    suffix: int = 0
    while True:
        try:
            # Strict rather than `exist_ok=True`, so the loser of a race
            # between two processes retries instead of clobbering.
            candidate.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            candidate = candidate.with_name(f"{stamp}-{suffix}")
        else:
            return candidate


def _reject_conflicting_outputs(load_run: Path | None, run_dir: Path | None) -> None:
    """Refuse the flag combination where one of the two could only be ignored.

    A reload writes back into the directory it read, so an output directory has
    nowhere to apply. A flag that silently does nothing inside a sweep script is
    the same class of bug as the clobber `--run-dir` exists to prevent.
    """
    if load_run is not None and run_dir is not None:
        raise ValueError("--run-dir has no meaning with --load-run")


def _save_run(
    g: RANModel,
    d: RANModel,
    history: dict[str, list[float]],
    params: EpochParams,
    *,
    batch_size: int,
    n_samples: int,
    dim: int,
    dataset: str,
    init_seed: int,
    data_seed: int,
    gaussian_params: GaussianConfig | None,
    variables: tuple[str, ...],
    hyperparameters: dict[str, Any],
    run_dir: Path | None = None,
) -> Path:
    with phase("save"):
        return _write_run_dir(
            g,
            d,
            history,
            params,
            batch_size=batch_size,
            n_samples=n_samples,
            dim=dim,
            dataset=dataset,
            init_seed=init_seed,
            data_seed=data_seed,
            gaussian_params=gaussian_params,
            variables=variables,
            hyperparameters=hyperparameters,
            run_dir=run_dir,
        )


def _write_run_dir(
    g: RANModel,
    d: RANModel,
    history: dict[str, list[float]],
    params: EpochParams,
    *,
    batch_size: int,
    n_samples: int,
    dim: int,
    dataset: str,
    init_seed: int,
    data_seed: int,
    gaussian_params: GaussianConfig | None,
    variables: tuple[str, ...],
    hyperparameters: dict[str, Any],
    run_dir: Path | None = None,
) -> Path:
    """Everything `_save_run` puts on disk. Split out only so the timer wraps a
    call rather than an indented body."""
    run_dir = _new_run_dir(run_dir)

    g.save(run_dir / "generator.keras")
    d.save(run_dir / "discriminator.keras")
    # Every epoch's parameters, not just the selected one's. `scan` already
    # emitted the stack; dropping it on the floor is what made re-scoring a run
    # under a different criterion cost a full retrain.
    _ = save_params(run_dir, params)
    np.savez(
        file=run_dir / "history.npz",
        # See ran.baselines.ibu: unpacking a str-keyed dict into savez means a
        # key could in principle be "allow_pickle", which is declared bool.
        **{k: np.array(object=v) for k, v in history.items()},  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    )

    # Every knob that distinguishes one run from another goes in here. A sweep
    # arm that is not recorded is not a measurement: `hidden_units`, `n_layers`
    # and `patience` were all absent from earlier configs despite changing the
    # run substantially.
    config_out: dict[str, Any] = {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "dim": dim,
        "dataset": dataset,
        "seed": init_seed,
        "data_seed": data_seed,
        **hyperparameters,
    }
    if dataset == "gaussian" and gaussian_params is not None:
        config_out["gaussian_params"] = gaussian_params.model_dump()
    else:
        config_out["variables"] = list(variables)
    json.dump(obj=config_out, fp=(run_dir / "config.json").open(mode="w"), indent=2)
    logger.info("Saved run to %s", run_dir)
    return run_dir


def _load_artifacts(run_dir: Path) -> tuple[RANModel, dict[str, list[float]]]:
    """Reload a finished run's generator and training history."""
    g: RANModel = keras.saving.load_model(run_dir / "generator.keras")
    history: dict[str, list[float]] = {
        k: v.tolist() for k, v in np.load(file=run_dir / "history.npz").items()
    }
    logger.info("Loaded run from %s", run_dir)
    return g, history


def _draw_figures(
    run_dir: Path,
    splits: DatasetSplits,
    g: RANModel,
    history: dict[str, list[float]],
    dim: int,
    var_info: list[VarInfo] | None,
    best_epoch: int,
    /,
    *,
    plots: bool,
) -> None:
    """Draw a run's figures, unless plots are turned off.

    Matplotlib is a large share of a short run's wall clock and none of it is
    needed to score one, so plots can be turned off for hyperparameter sweeps,
    bootstrapping, etc. The artifacts are already on disk by then, so
    `--load-run` on the same directory draws them later.

    The guard lives here rather than at the call site because the IBU overlay is
    part of the same decision: `_load_baseline_weights` exists only to feed
    these three calls.

    `plot_selection` is skipped -- rather than left to raise `KeyError` -- when
    `val_mmd` is absent from `history`: a run saved before this branch has no
    MMD columns at all, and `--load-run` must still be able to replot it. This
    is the same treatment `CLAUDE.md` already documents for the `val_g` column
    that `plot_losses` deliberately never reads.
    """
    if not plots:
        return
    ibu_weights: list[EventArray] | None = _load_baseline_weights(run_dir, dim)
    plot_levels(
        splits.test,
        g,
        detector_path=run_dir / "detector_level.pdf",
        particle_path=run_dir / "particle_level.pdf",
        var_info=var_info,
        ibu_weights=ibu_weights,
    )
    plot_losses(history, save_path=run_dir / "losses.pdf")
    if "val_mmd" in history:
        plot_selection(history, best_epoch, save_path=run_dir / "selection.pdf")
    else:
        logger.debug("No val_mmd in history, skipping selection.pdf")


def _load_baseline_weights(
    run_dir: Path,
    dim: int,
) -> list[EventArray] | None:
    """Pick up IBU weights from the run dir, if that baseline has run."""
    ibu_weights: list[EventArray] | None = None
    ibu_path: Path = run_dir / "ibu_weights.npz"
    if ibu_path.exists():
        ibu_data: dict[str, Any] = np.load(ibu_path)
        ibu_weights = [ibu_data[f"weights_{i}"] for i in range(dim)]
        logger.info("Loaded IBU weights from %s", ibu_path)
    return ibu_weights


def _particle_curve(
    splits: DatasetSplits,
    result: TrainResult,
) -> tuple[list[float], tuple[float, ...]] | None:
    """Particle-level MMD per epoch: the diagnostic, never the criterion.

    Returns `None` for a real measurement, which has no truth to score
    against. Selection has already happened by the time this runs, so nothing
    the generator saw depends on it.

    Unlike `train.py`'s detector-level selection, calling `.partition()` here
    is correct: this runs outside the trace, after selection, and needs the
    answer key `train.py` must never see.
    """
    pops: Populations = splits.val.as_arrays().partition()
    if not pops.has_truth:
        return None
    z_true: EventArray = pops.require_truth()
    z_gen: EventArray = pops.mc.z
    # Seeded off `splits.train.seed` (`data_seed`), the way `train.py`'s own
    # detector-level draws are: `s`/`s+1` val-detector and `s+2`/`s+3`
    # test-detector are already spoken for, so this uses `s+4`/`s+5`.
    seed: int = splits.train.seed
    i_t = subsample_indices(seed + 4, z_true.shape[0], MMD_SUBSAMPLE)
    i_g = subsample_indices(seed + 5, z_gen.shape[0], MMD_SUBSAMPLE)
    z_gen_sub: EventArray = z_gen[i_g]
    ref, comp = jnp.asarray(z_true[i_t]), jnp.asarray(z_gen_sub)
    sigmas: tuple[float, ...] = bandwidths(ref)
    curve, _ = mmd_curve(
        build_cache(ref, comp, sigmas=sigmas),
        _weights_per_epoch(result.g, result.params, z_gen_sub),
    )
    return curve.tolist(), sigmas


def _finish_run(
    splits: DatasetSplits,
    result: TrainResult,
    /,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Merge the particle diagnostic in, and assemble what gets recorded."""
    history: dict[str, list[float]] = dict(result.history)
    particle = _particle_curve(splits, result)
    sigmas_particle: tuple[float, ...] = ()
    if particle is not None:
        history["val_mmd_particle"], sigmas_particle = particle
    return history, {
        "mmd_subsample": MMD_SUBSAMPLE,
        "mmd_sigmas_detector": list(result.sigmas),
        "mmd_sigmas_particle": list(sigmas_particle),
        "mmd_test": result.mmd_test,
        "best_epoch": result.best_epoch,
    }


def run(
    batch_size: int,
    n_samples: int,
    config: Path | None,
    dataset: DatasetName,
    variables: tuple[str, ...],
    load_run: Path | None,
    hidden_units: int,
    n_layers: int,
    seed: int | None,
    data_seed: int,
    *,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 3e-5,
    lr_d: float = 1e-4,
    lambda_dispersion: float = 0.015,
    plots: bool = True,
    run_dir: Path | None = None,
) -> None:
    """Train (or reload) one run, then report where its wall clock went.

    The timing report is in a `finally` because a run that fell over is exactly
    the one whose breakdown is worth having --- the phase that raised is
    recorded with the time it burned before it did.

    `timings.json` needs a directory to land in. `--run-dir` and `--load-run`
    both name one up front, so a crash there still gets a file; a fresh run
    under the default timestamp has no directory until `_save_run` makes one,
    and if it dies first the table on stderr is all there is.
    """
    _reject_conflicting_outputs(load_run, run_dir)

    written_to: Path | None = load_run or run_dir
    try:
        written_to = _pipeline(
            batch_size,
            n_samples,
            config,
            dataset,
            variables,
            load_run,
            hidden_units,
            n_layers,
            seed,
            data_seed,
            n_epochs=n_epochs,
            n_disc_steps=n_disc_steps,
            lr_g=lr_g,
            lr_d=lr_d,
            lambda_dispersion=lambda_dispersion,
            plots=plots,
            run_dir=run_dir,
        )
    finally:
        report()
        if written_to is not None:
            write(written_to)


def _pipeline(
    batch_size: int,
    n_samples: int,
    config: Path | None,
    dataset: DatasetName,
    variables: tuple[str, ...],
    load_run: Path | None,
    hidden_units: int,
    n_layers: int,
    seed: int | None,
    data_seed: int,
    *,
    n_epochs: int,
    n_disc_steps: int,
    lr_g: float,
    lr_d: float,
    lambda_dispersion: float,
    plots: bool,
    run_dir: Path | None,
) -> Path:
    """The run itself, returning the directory its artifacts landed in."""
    # Each dataset fills in only its own metadata, but the plots and the saved
    # config are handed both, so the other one has to exist as None.
    gaussian_params: GaussianConfig | None = None
    var_info: list[VarInfo] | None = None

    # When loading a saved run, read config from that run and rebuild the
    # Gaussian params it recorded, so a reload never re-parses --config (which
    # may not even be passed, or may have since changed on disk).
    saved_gaussian_config: GaussianConfig | None = None
    # No `TrainResult` on the reload path, so `best_epoch` has to come from
    # what training recorded. Absent on a run saved before this branch, same
    # as `val_mmd`/`val_ess` themselves -- see R15 in the task brief.
    saved_best_epoch: int = -1
    if load_run is not None:
        run_dir = Path(load_run)
        # parse_run_config validates an already-decoded JSON object, not text --
        # the IBU baseline that also calls it json.loads first.
        saved_config: RunConfig = parse_run_config(
            raw=json.loads(s=(run_dir / "config.json").read_text())
        )
        dataset: DatasetName = saved_config.dataset
        n_samples: int = saved_config.n_samples
        batch_size: int = saved_config.batch_size
        dim: int = saved_config.dim
        # Runs predating seed recording used the then-hardcoded default of 42.
        data_seed: int = saved_config.data_seed
        saved_best_epoch = saved_config.source.get("best_epoch", -1)
        if dataset == DatasetName.jets:
            variables = tuple(saved_config.variable_names)
        else:
            saved_gaussian_config = gaussian_config_from_run_config(
                saved_config.source["gaussian_params"], dim
            )

    with phase("data"):
        # Which branch this took --- cache hit, generated, downloaded --- is
        # filled in from inside the loaders, which know and this does not.
        if dataset == DatasetName.gaussian:
            splits, dim, gaussian_params = _prepare_gaussian(
                config,
                saved_gaussian_config,
                batch_size,
                n_samples,
                data_seed,
            )
        elif dataset == DatasetName.jets:
            splits, dim, var_info = _prepare_jets(
                n_samples, batch_size, variables, data_seed
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")

    g: RANModel
    history: dict[str, list[float]]
    best_epoch: int
    if load_run is not None:
        run_dir = Path(load_run)
        with phase("load"):
            g, history = _load_artifacts(run_dir)
        best_epoch = saved_best_epoch
    else:
        with phase("train"):
            result: TrainResult = train(
                splits,
                dim,
                hidden_units,
                n_layers,
                seed,
                n_epochs=n_epochs,
                n_disc_steps=n_disc_steps,
                lr_g=lr_g,
                lr_d=lr_d,
                lambda_dispersion=lambda_dispersion,
            )
        g = result.g
        best_epoch = result.best_epoch
        with phase("particle_mmd"):
            history, mmd_record = _finish_run(splits, result)
        run_dir = _save_run(
            result.g,
            result.d,
            history,
            result.params,
            batch_size=batch_size,
            n_samples=n_samples,
            dim=dim,
            dataset=dataset,
            init_seed=result.seed,
            data_seed=data_seed,
            gaussian_params=gaussian_params,
            variables=variables,
            hyperparameters={
                "hidden_units": hidden_units,
                "n_layers": n_layers,
                "n_epochs": n_epochs,
                "n_disc_steps": n_disc_steps,
                "lr_g": lr_g,
                "lr_d": lr_d,
                "lambda_dispersion": lambda_dispersion,
                **mmd_record,
            },
            run_dir=run_dir,
        )

    with phase("plots"):
        _draw_figures(
            run_dir, splits, g, history, dim, var_info, best_epoch, plots=plots
        )

    # Metrics (run last so failures don't block plots/checkpoints)
    with phase("evaluate"):
        try:
            _ = evaluate_run(run_dir, force=(load_run is None))
        except Exception:
            logger.exception(msg="Metric evaluation failed")

    return run_dir
