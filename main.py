from datetime import datetime, timezone
from pathlib import Path
import fire

from datasets import DatasetSplits, RAN_Dataset
from train import train
from plotting import plot_detector_level, plot_particle_level, plot_losses

import keras
import numpy as np


def main(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    smearing: float = 0.5,
    dim: int = 1,
) -> None:
    """
    Main entry point.
    """

    splits: DatasetSplits = RAN_Dataset(
        batch_size=batch_size
        ).generate_gaussian_dataset(
        n_samples=n_samples,
        smearing=smearing,
        dim=dim,
    )
    g: keras.Model
    history: dict[str, list[float | np.floating]]
    g, _, history = train(splits, dim=dim)

    run_dir: Path = Path("plots") / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_detector_level(splits.test, g, save_path=run_dir / "detector_level.pdf")
    plot_particle_level(splits.test, g, save_path=run_dir / "particle_level.pdf")
    plot_losses(history, save_path=run_dir / "losses.pdf")


if __name__ == "__main__":
    fire.Fire(main)
