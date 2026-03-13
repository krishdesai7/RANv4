from datasets import DatasetSplits
from train import train
from plotting import plot_detector_level, plot_particle_level, plot_losses

import keras
import numpy as np


def main() -> None:
    """
    Main entry point.
    """
    g: keras.Model
    splits: DatasetSplits
    history: dict[str, list[float | np.floating]]
    g, _, splits, history = train()
    plot_detector_level(splits.test, g)
    plot_particle_level(splits.test, g)
    plot_losses(history)


if __name__ == "__main__":
    main()
