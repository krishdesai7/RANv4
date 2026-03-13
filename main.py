from train import train
from plotting import plot_detector_level, plot_particle_level, plot_losses


def main():
    g, d, splits, history = train()
    plot_detector_level(splits.test, g)
    plot_particle_level(splits.test, g)
    plot_losses(history)


if __name__ == "__main__":
    main()
