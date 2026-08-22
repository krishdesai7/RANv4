from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
from .plotting import plot_detector_level, plot_losses, plot_particle_level
from .rantypes import JET_OBS, DatasetName, GaussianConfig, VarInfo
from .train import train

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any

    from numpy._typing import _DTypeLike
    from numpy.typing import NDArray

    from .rantypes import DatasetSplits, RANModel, RunConfig

logger: Logger = logging.getLogger(__name__)


def _prepare_gaussian[T: np.floating](
    config: Path | None,
    saved_config: GaussianConfig | None,
    batch_size: int,
    n_samples: int,
    data_seed: int,
    *,
    dtype: _DTypeLike[T],
) -> tuple[DatasetSplits[T], int, GaussianConfig]:
    builder: RANDataset[T] = RANDataset(
        batch_size=batch_size, seed=data_seed, dtype=dtype
    )
    if saved_config is not None:
        gaussian_params: GaussianConfig = saved_config
        # Reload: use stored params from config.json
        splits: DatasetSplits[T] = builder.generate_gaussian_dataset(
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


def _prepare_jets[T: np.floating](
    n_samples: int,
    batch_size: int,
    variables: frozenset[str],
    data_seed: int,
    *,
    dtype: _DTypeLike[T],
) -> tuple[DatasetSplits[T], int, list[VarInfo]]:
    """Build jet splits plus the per-variable metadata the plots need."""
    std_params: dict[str, tuple[T, T]]
    splits, dim, std_params = load_jet_dataset(
        n_samples=n_samples,
        batch_size=batch_size,
        variables=variables,
        seed=data_seed,
        dtype=dtype,
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


def _save_run(
    g: RANModel,
    d: RANModel,
    history: dict[str, list[float]],
    *,
    batch_size: int,
    n_samples: int,
    dim: int,
    dataset: str,
    init_seed: int,
    data_seed: int,
    gaussian_params: GaussianConfig | None,
    variables: frozenset[str],
) -> Path:
    run_dir: Path = Path("runs") / datetime.now(tz=UTC).strftime(
        format="%Y-%m-%dT%H%M%SZ"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    g.save(run_dir / "generator.keras")
    d.save(run_dir / "discriminator.keras")
    np.savez(
        file=run_dir / "history.npz",
        # See ran.baselines.ibu: unpacking a str-keyed dict into savez means a
        # key could in principle be "allow_pickle", which is declared bool.
        **{k: np.array(object=v) for k, v in history.items()},  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    )

    config_out: dict[str, Any] = {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "dim": dim,
        "dataset": dataset,
        "seed": init_seed,
        "data_seed": data_seed,
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


def _load_baseline_weights[T: np.floating = np.double](
    run_dir: Path,
    dim: int,
) -> list[NDArray[T]] | None:
    """Pick up IBU weights from the run dir, if those baselines have run."""
    ibu_weights: list[NDArray[T]] | None = None
    ibu_path: Path = run_dir / "ibu_weights.npz"
    if ibu_path.exists():
        ibu_data: dict[str, Any] = np.load(ibu_path)
        ibu_weights = [ibu_data[f"weights_{i}"] for i in range(dim)]
        logger.info("Loaded IBU weights from %s", ibu_path)
    return ibu_weights


def run(
    batch_size: int,
    n_samples: int,
    config: Path | None,
    dataset: DatasetName,
    variables: frozenset[str],
    load_run: Path | None,
    hidden_units: int,
    n_layers: int,
    patience: int,
    seed: int | None,
    data_seed: int,
) -> None:
    # Each dataset fills in only its own metadata, but the plots and the saved
    # config are handed both, so the other one has to exist as None.
    gaussian_params: GaussianConfig | None = None
    var_info: list[VarInfo] | None = None

    # When loading a saved run, read config from that run and rebuild the
    # Gaussian params it recorded, so a reload never re-parses --config (which
    # may not even be passed, or may have since changed on disk).
    saved_gaussian_config: GaussianConfig | None = None
    if load_run is not None:
        run_dir = Path(load_run)
        # parse_run_config validates an already-decoded JSON object, not text --
        # the two baselines that call it both json.loads first.
        saved_config: RunConfig = parse_run_config(
            raw=json.loads(s=(run_dir / "config.json").read_text())
        )
        dataset: DatasetName = saved_config.dataset
        n_samples: int = saved_config.n_samples
        batch_size: int = saved_config.batch_size
        dim: int = saved_config.dim
        # Runs predating seed recording used the then-hardcoded default of 42.
        data_seed: int = saved_config.data_seed
        if dataset == DatasetName.jets:
            variables = frozenset(saved_config.variable_names)
        else:
            saved_gaussian_config = gaussian_config_from_run_config(
                saved_config.source["gaussian_params"], dim
            )

    if dataset == DatasetName.gaussian:
        splits, dim, gaussian_params = _prepare_gaussian(
            config,
            saved_gaussian_config,
            batch_size,
            n_samples,
            data_seed,
            dtype=np.double,
        )
    elif dataset == DatasetName.jets:
        splits, dim, var_info = _prepare_jets(
            n_samples, batch_size, variables, data_seed, dtype=np.double
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    g: RANModel
    history: dict[str, list[float]]
    if load_run is not None:
        run_dir = Path(load_run)
        g, history = _load_artifacts(run_dir)
    else:
        d: RANModel
        init_seed: int
        g, d, history, init_seed = train(
            splits,
            dim,
            hidden_units,
            n_layers,
            seed,
            patience,
        )
        run_dir: Path = _save_run(
            g,
            d,
            history,
            batch_size=batch_size,
            n_samples=n_samples,
            dim=dim,
            dataset=dataset,
            init_seed=init_seed,
            data_seed=data_seed,
            gaussian_params=gaussian_params,
            variables=variables,
        )

    ibu_weights = _load_baseline_weights(run_dir, dim)

    # Plots
    plot_detector_level(
        splits.test,
        g,
        save_path=run_dir / "detector_level.pdf",
        var_info=var_info,
        ibu_weights=ibu_weights,
    )
    plot_particle_level(
        splits.test,
        g,
        save_path=run_dir / "particle_level.pdf",
        var_info=var_info,
        ibu_weights=ibu_weights,
    )
    plot_losses(history, save_path=run_dir / "losses.pdf")

    # Metrics (run last so failures don't block plots/checkpoints)
    try:
        evaluate_run(run_dir, force=(load_run is None))
    except Exception:
        logger.exception(msg="Metric evaluation failed")
