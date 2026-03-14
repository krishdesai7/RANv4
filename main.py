from datetime import datetime, timezone
from pathlib import Path
import json
import fire

from datasets import DatasetSplits, RAN_Dataset
from train import train
from plotting import (
    plot_detector_level, plot_particle_level, plot_losses,
    _collect_data, _get_weights,
)

import keras
import numpy as np
from scipy.stats import wasserstein_distance_nd


def _evaluate_wasserstein(
    test_dataset,
    g: keras.Model,
) -> dict[str, float]:
    """Compute Wasserstein distances before and after reweighting."""
    z, x, y = _collect_data(test_dataset)
    w: np.ndarray = _get_weights(g, z[y == 0])

    wd: dict[str, float] = {
        "detector_before": wasserstein_distance_nd(x[y == 1], x[y == 0]),
        "detector_after":  wasserstein_distance_nd(x[y == 1], x[y == 0], v_weights=w),
        "particle_before": wasserstein_distance_nd(z[y == 1], z[y == 0]),
        "particle_after":  wasserstein_distance_nd(z[y == 1], z[y == 0], v_weights=w),
    }
    return wd


def _print_wasserstein(wd: dict[str, float]) -> None:
    print("\nWasserstein Distance:")
    for level in ("detector", "particle"):
        before: float = wd[f"{level}_before"]
        after: float = wd[f"{level}_after"]
        improvement: float = (1 - after / before) * 100 if before > 0 else 0.0
        print(f"  {level:>8s}:  {before:.4f} -> {after:.4f}  ({improvement:+.1f}%)")


def main(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    smearing: float = 0.5,
    dim: int = 1,
    load_run: str | None = None,
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

    if load_run is not None:
        run_dir = Path(load_run)
        g: keras.Model = keras.saving.load_model(run_dir / "generator.keras")
        history: dict[str, list] = {
            k: v.tolist() for k, v in np.load(run_dir / "history.npz").items()
        }
        print(f"Loaded run from {run_dir}")
    else:
        g, d, history = train(splits, dim=dim)

        run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=True)

        g.save(run_dir / "generator.keras")
        d.save(run_dir / "discriminator.keras")
        np.savez(run_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})
        json.dump(
            {"batch_size": batch_size, "n_samples": n_samples,
             "smearing": smearing, "dim": dim},
            (run_dir / "config.json").open("w"), indent=2,
        )
        print(f"Saved run to {run_dir}")

    # Wasserstein validation
    wd: dict[str, float] = _evaluate_wasserstein(splits.test, g)
    _print_wasserstein(wd)
    json.dump(wd, (run_dir / "wasserstein.json").open("w"), indent=2)

    # Plots
    plot_detector_level(splits.test, g, save_path=run_dir / "detector_level.pdf")
    plot_particle_level(splits.test, g, save_path=run_dir / "particle_level.pdf")
    plot_losses(history, save_path=run_dir / "losses.pdf")


if __name__ == "__main__":
    fire.Fire(main)
