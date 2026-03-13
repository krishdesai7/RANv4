from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetSplits
from train import train
from plotting import plot_detector_level, plot_particle_level, plot_losses

import keras
import numpy as np


def main() -> None:
    """
    Main entry point.
    """
    run_dir: Path = Path("plots") / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    g: keras.Model
    splits: DatasetSplits
    history: dict[str, list[float | np.floating]]
    g, _, splits, history = train()
    plot_detector_level(splits.test, g, save_path=run_dir / "detector_level.pdf")
    plot_particle_level(splits.test, g, save_path=run_dir / "particle_level.pdf")
    plot_losses(history, save_path=run_dir / "losses.pdf")


if __name__ == "__main__":
    main()
