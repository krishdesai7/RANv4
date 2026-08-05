import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import numpy.typing as npt

from ran.data import JET_OBS, DatasetSplits, RANDataset, load_jet_dataset
from ran.data.config import parse_gaussian_config
from ran.evaluate import evaluate_run
from ran.plotting import (
    VarInfo,
    plot_detector_level,
    plot_losses,
    plot_particle_level,
)
from ran.train import train

logger = logging.getLogger(__name__)


def _prepare_gaussian(
    config: str | None,
    saved_config: dict[str, Any],
    batch_size: int,
    n_samples: int,
    data_seed: int,
) -> tuple[DatasetSplits, int, dict[str, Any]]:
    """Build Gaussian splits from a reloaded run's config, or from a YAML file.

    Returns the splits, the dimensionality, and the parsed Gaussian params --
    the last so a fresh run can record them in its own config.json.
    """
    builder = RANDataset(batch_size=batch_size, seed=data_seed)
    gaussian_params: dict[str, Any]
    if saved_config:
        # Reload: use stored params from config.json
        gaussian_params = saved_config["gaussian_params"]
        raw_params: dict[str, Any] = {
            k: v for k, v in gaussian_params.items() if k != "dim"
        }
        splits = builder.generate_gaussian_dataset(
            params=raw_params, n_samples=n_samples
        )
    else:
        # Fresh run: parse YAML config
        if config is None:
            raise ValueError("Gaussian mode requires --config path/to/config.yaml")
        gaussian_params = parse_gaussian_config(config)
        splits = builder.generate_gaussian_dataset(
            config_path=config, n_samples=n_samples
        )
    return splits, gaussian_params["dim"], gaussian_params


def _prepare_jets(
    n_samples: int,
    batch_size: int,
    variables: tuple[str, ...],
    data_seed: int,
) -> tuple[DatasetSplits, int, list[VarInfo]]:
    """Build jet splits plus the per-variable metadata the plots need."""
    std_params: dict[str, tuple[np.double, np.double]]
    splits, dim, std_params = load_jet_dataset(
        n_samples=n_samples,
        batch_size=batch_size,
        variables=variables,
        seed=data_seed,
    )
    var_info = [
        VarInfo(
            xlim=JET_OBS[v]["xlim"],
            xlabel=JET_OBS[v]["xlabel"],
            symbol=JET_OBS[v]["symbol"],
            mu=std_params[v][0],
            sigma=std_params[v][1],
        )
        for v in variables
    ]
    return splits, dim, var_info


def _to_list(v: Any) -> Any:
    return v.tolist() if hasattr(v, "tolist") else v


def _save_run(
    g: keras.Model,
    d: keras.Model,
    history: dict[str, list[float]],
    *,
    batch_size: int,
    n_samples: int,
    dim: int,
    dataset: str,
    init_seed: int,
    data_seed: int,
    gaussian_params: dict[str, Any] | None,
    variables: tuple[str, ...],
) -> Path:
    """Write models, history and config to a fresh timestamped run directory.

    Gaussian params are stored as covariance matrices so runs are
    self-contained and reloadable without the original YAML. `init_seed` is the
    resolved weight-init seed, never None, so a run drawn from entropy is still
    reproducible via --seed after the fact.
    """
    run_dir: Path = Path("runs") / datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    g.save(run_dir / "generator.keras")
    d.save(run_dir / "discriminator.keras")
    np.savez(
        run_dir / "history.npz",
        # See ran.baselines.ibu: unpacking a str-keyed dict into savez means a
        # key could in principle be "allow_pickle", which is declared bool.
        **{k: np.array(v) for k, v in history.items()},  # pyrefly: ignore[bad-argument-type]
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
        config_out["gaussian_params"] = {
            "dim": dim,
            "mu_gen": _to_list(gaussian_params["mu_gen"]),
            "mu_true": _to_list(gaussian_params["mu_true"]),
            "sigma_gen": _to_list(gaussian_params["cov_gen"]),
            "sigma_true": _to_list(gaussian_params["cov_true"]),
            "sigma_detector": _to_list(gaussian_params["cov_detector"]),
        }
    else:
        config_out["variables"] = list(variables)
    json.dump(config_out, (run_dir / "config.json").open("w"), indent=2)
    logger.info("Saved run to %s", run_dir)
    return run_dir


def _load_baseline_weights(
    run_dir: Path, dim: int
) -> tuple[npt.NDArray[np.double] | None, list[npt.NDArray[np.double]] | None]:
    """Pick up OmniFold/IBU weights from the run dir, if those baselines have run."""
    omnifold_weights = None
    ibu_weights: list[npt.NDArray[np.double]] | None = None
    of_path: Path = run_dir / "omnifold_weights.npz"
    ibu_path: Path = run_dir / "ibu_weights.npz"
    if of_path.exists():
        omnifold_weights = np.load(of_path)["weights"]
        logger.info("Loaded OmniFold weights from %s", of_path)
    if ibu_path.exists():
        ibu_data: dict[str, Any] = np.load(ibu_path)
        ibu_weights = [ibu_data[f"weights_{i}"] for i in range(dim)]
        logger.info("Loaded IBU weights from %s", ibu_path)
    return omnifold_weights, ibu_weights


def run(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    config: str | None = None,
    dataset: str = "gaussian",
    variables: tuple[str, ...] = ("m", "M", "w", "tau21", "zg", "sdm"),
    load_run: str | None = None,
    hidden_units: int = 64,
    n_layers: int = 2,
    patience: int = 5,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
    """
    Main entry point.

    Arguments:
        seed: Weight-initialization seed. Omit to draw one from system entropy;
            the value used is always recorded in config.json, so any run can be
            reproduced afterwards. Vary this across runs (holding data_seed
            fixed) to build an ensemble for model-uncertainty bands.
        data_seed: Dataset seed, controlling generation, the shuffle, the
            train/val/test split and the batch order. Keep fixed across an
            ensemble so replicas differ only in initialization.
    """
    run_dir: Path
    splits: DatasetSplits
    var_info: list[VarInfo] | None = None
    gaussian_params: dict | None = None
    dim: int = 1

    # When loading a saved run, read config from that run. Defaults to {} so
    # every branch below can read it unconditionally.
    saved_config: dict[str, Any] = {}
    if load_run is not None:
        run_dir = Path(load_run)
        saved_config = json.loads((run_dir / "config.json").read_text())
        dataset = saved_config.get("dataset", "gaussian")
        n_samples = saved_config["n_samples"]
        batch_size = saved_config["batch_size"]
        dim = saved_config["dim"]
        # Runs predating seed recording used the then-hardcoded default of 42.
        data_seed = saved_config.get("data_seed", 42)
        if dataset == "jets":
            variables = tuple(saved_config["variables"])

    if dataset == "gaussian":
        splits, dim, gaussian_params = _prepare_gaussian(
            config, saved_config, batch_size, n_samples, data_seed
        )
    elif dataset == "jets":
        splits, dim, var_info = _prepare_jets(
            n_samples, batch_size, variables, data_seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if load_run is not None:
        run_dir = Path(load_run)
        g: keras.Model = keras.saving.load_model(run_dir / "generator.keras")
        history: dict[str, list[float]] = {
            k: v.tolist() for k, v in np.load(run_dir / "history.npz").items()
        }
        logger.info("Loaded run from %s", run_dir)
    else:
        d: keras.Model
        init_seed: int
        g, d, history, init_seed = train(
            splits,
            dim=dim,
            hidden_units=hidden_units,
            n_layers=n_layers,
            patience=patience,
            seed=seed,
        )
        run_dir = _save_run(
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

    omnifold_weights, ibu_weights = _load_baseline_weights(run_dir, dim)

    # Plots
    plot_detector_level(
        splits.test,
        g,
        save_path=run_dir / "detector_level.pdf",
        var_info=var_info,
        omnifold_weights=omnifold_weights,
        ibu_weights=ibu_weights,
    )
    plot_particle_level(
        splits.test,
        g,
        save_path=run_dir / "particle_level.pdf",
        var_info=var_info,
        omnifold_weights=omnifold_weights,
        ibu_weights=ibu_weights,
    )
    plot_losses(history, save_path=run_dir / "losses.pdf")

    # Metrics (run last so failures don't block plots/checkpoints)
    try:
        evaluate_run(run_dir, force=(load_run is None))
    except Exception:
        logger.exception("Metric evaluation failed")
