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
    smearing: float = 0.5,
    dim: int = 1,
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
    # When loading a saved run, read config from that run
    if load_run is not None:
        run_dir = Path(load_run)
        config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
        dataset = config.get("dataset", "gaussian")
        n_samples: int = config["n_samples"]
        batch_size: int = config["batch_size"]
        dim: int = config["dim"]
        if dataset == "gaussian":
            smearing: float = config.get("smearing", smearing)
        else:
            variables = tuple(config["variables"])


    if dataset == "gaussian":
        splits = RAN_Dataset(
            batch_size=batch_size
            ).generate_gaussian_dataset(
            n_samples=n_samples,
            smearing=smearing,
            dim=dim,
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

        run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=True)

        g.save(run_dir / "generator.keras")
        d.save(run_dir / "discriminator.keras")
        np.savez(run_dir / "history.npz", **{k: np.array(v) for k, v in history.items()}) # type: ignore
        config = {"batch_size": batch_size, "n_samples": n_samples,
                  "dim": dim, "dataset": dataset}
        if dataset == "gaussian":
            config["smearing"] = smearing
        else:
            config["variables"] = list(variables)
        json.dump(config, (run_dir / "config.json").open("w"), indent=2)
        print(f"Saved run to {run_dir}")

    # Plots
    plot_detector_level(splits.test, g, save_path=run_dir / "detector_level.pdf",
                        var_info=var_info)
    plot_particle_level(splits.test, g, save_path=run_dir / "particle_level.pdf",
                        var_info=var_info)
    plot_losses(history, save_path=run_dir / "losses.pdf")

    # Metrics (run last so failures don't block plots/checkpoints)
    try:
        evaluate_run(run_dir, force=True)
    except Exception as e:
        print(f"Metric evaluation failed: {e}")


if __name__ == "__main__":
    fire.Fire(main)
