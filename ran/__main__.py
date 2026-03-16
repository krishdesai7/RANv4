import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import fire

from ran.data import DatasetSplits, RAN_Dataset, load_jet_dataset, JET_OBS
from ran.evaluate import evaluate_run

from ran.train import train
from ran.plotting import (
    plot_detector_level,
    plot_particle_level,
    plot_losses,
    VarInfo,
)

import keras
import numpy as np


def main(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    config: str | None = None,
    dataset: str = "gaussian",
    variables: tuple[str, ...] = ("m", "M", "w", "tau21", "zg", "sdm"),
    load_run: str | None = None,
) -> None:
    """
    Main entry point.
    """
    run_dir: Path
    splits: DatasetSplits
    var_info: list[VarInfo] | None = None
    gaussian_params: dict | None = None
    dim: int = 1

    # When loading a saved run, read config from that run
    if load_run is not None:
        run_dir = Path(load_run)
        saved_config: dict[str, Any] = json.loads(
            (run_dir / "config.json").read_text()
        )
        dataset = saved_config.get("dataset", "gaussian")
        n_samples = saved_config["n_samples"]
        batch_size = saved_config["batch_size"]
        dim = saved_config["dim"]
        if dataset == "jets":
            variables = tuple(saved_config["variables"])

    if dataset == "gaussian":
        if load_run is not None:
            # Reload: use stored params from config.json
            gaussian_params = saved_config["gaussian_params"]
            dim = gaussian_params["dim"]
            raw_params = {k: v for k, v in gaussian_params.items() if k != "dim"}
            splits = RAN_Dataset(
                batch_size=batch_size
            ).generate_gaussian_dataset(
                params=raw_params,
                n_samples=n_samples,
            )
        else:
            # Fresh run: parse YAML config
            if config is None:
                raise ValueError(
                    "Gaussian mode requires --config path/to/config.yaml"
                )
            from ran.data.config import parse_gaussian_config
            gaussian_params = parse_gaussian_config(config)
            dim = gaussian_params["dim"]
            splits = RAN_Dataset(
                batch_size=batch_size
            ).generate_gaussian_dataset(
                config_path=config,
                n_samples=n_samples,
            )
    elif dataset == "jets":
        std_params: dict[str, tuple[np.double, np.double]]
        splits, dim, std_params = load_jet_dataset(
            n_samples=n_samples,
            batch_size=batch_size,
            variables=variables,
        )
        var_info = [
            VarInfo(
                xlim=JET_OBS[v]["xlim"],
                xlabel=JET_OBS[v]["xlabel"],
                symbol=JET_OBS[v]["symbol"],
                mu=std_params[v][0],
                sigma=std_params[v][1],
            ) for v in variables
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if load_run is not None:
        run_dir = Path(load_run)
        g: keras.Model = keras.saving.load_model(run_dir / "generator.keras")
        history: dict[str, list] = {
            k: v.tolist() for k, v in np.load(run_dir / "history.npz").items()
        }
        print(f"Loaded run from {run_dir}")
    else:
        d: keras.Model
        g, d, history = train(splits, dim=dim)

        run_dir = Path("runs") / datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H%M%SZ"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        g.save(run_dir / "generator.keras")
        d.save(run_dir / "discriminator.keras")
        np.savez(
            run_dir / "history.npz",
            **{k: np.array(v) for k, v in history.items()},  # type: ignore
        )

        # Save config — Gaussian params stored as covariance matrices
        # so runs are self-contained and reloadable without original YAML
        config_out: dict[str, Any] = {
            "batch_size": batch_size,
            "n_samples": n_samples,
            "dim": dim,
            "dataset": dataset,
        }
        if dataset == "gaussian" and gaussian_params is not None:
            def _to_list(v):
                return v.tolist() if hasattr(v, "tolist") else v

            config_out["gaussian_params"] = {
                "dim": dim,
                "mu_mc": _to_list(gaussian_params["mu_mc"]),
                "mu_true": _to_list(gaussian_params["mu_true"]),
                "sigma_mc": _to_list(gaussian_params["cov_mc"]),
                "sigma_true": _to_list(gaussian_params["cov_true"]),
                "sigma_detector": _to_list(gaussian_params["cov_detector"]),
            }
        else:
            config_out["variables"] = list(variables)
        json.dump(config_out, (run_dir / "config.json").open("w"), indent=2)
        print(f"Saved run to {run_dir}")

    # Plots
    plot_detector_level(
        splits.test, g, save_path=run_dir / "detector_level.pdf",
        var_info=var_info,
    )
    plot_particle_level(
        splits.test, g, save_path=run_dir / "particle_level.pdf",
        var_info=var_info,
    )
    plot_losses(history, save_path=run_dir / "losses.pdf")

    # Metrics (run last so failures don't block plots/checkpoints)
    try:
        evaluate_run(run_dir, force=True)
    except Exception as e:
        print(f"Metric evaluation failed: {e}")


if __name__ == "__main__":
    fire.Fire(main)
