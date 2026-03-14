"""Load jet substructure data for RAN training.

Checks .cache/ for per-variable .npz files. If missing, invokes
download_jet_data to fetch from Zenodo. Loads, subsamples, z-score
standardizes (using MC gen-level statistics only), and builds a
tf.data.Dataset via RAN_Dataset.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from datasets import RAN_Dataset, DatasetSplits

SUBSTRUCTURE_VARIABLES = ("m", "M", "w", "tau21", "zg", "sdm")
CACHE_DIR = Path(".cache")


def load_jet_dataset(
    n_samples: int = 500_000,
    batch_size: int = 1024,
    cache_dir: Path = CACHE_DIR,
) -> tuple[DatasetSplits, int]:
    """Load jet substructure data and return DatasetSplits.

    Each of the 6 substructure variables is z-score standardized using
    the MC gen-level (z_gen) mean and std. The same parameters are applied
    to all four arrays (z_true, x_data, z_gen, x_sim) to avoid information
    leakage and preserve correlations.

    Args:
        n_samples: Number of events to use per class (data and MC).
        batch_size: Batch size for tf.data.Dataset.
        cache_dir: Directory containing per-variable .npz files.

    Returns:
        (splits, dim): DatasetSplits and feature dimensionality.
    """
    # Check cache, download if needed
    missing = [v for v in SUBSTRUCTURE_VARIABLES
               if not (cache_dir / f"{v}.npz").exists()]
    if missing:
        from download_jet_data import download_jet_data
        print("Cached jet data not found. Downloading from Zenodo...")
        download_jet_data(cache_dir)

    n_features: int = len(SUBSTRUCTURE_VARIABLES)

    # Check available samples
    with np.load(cache_dir / f"{SUBSTRUCTURE_VARIABLES[0]}.npz") as f:
        n_avail: int = min(len(f["z_true"]), len(f["z_gen"]))
    if n_samples > n_avail:
        raise ValueError(
            f"Requested {n_samples} samples but only {n_avail} available"
        )

    # Initialize arrays
    z_true = np.empty((n_samples, n_features), dtype=np.float64)
    x_data = np.empty((n_samples, n_features), dtype=np.float64)
    z_gen = np.empty((n_samples, n_features), dtype=np.float64)
    x_sim = np.empty((n_samples, n_features), dtype=np.float64)

    # Load, subsample, and standardize each variable
    for i, var in enumerate(SUBSTRUCTURE_VARIABLES):
        with np.load(cache_dir / f"{var}.npz") as f:
            z_true[:, i] = f["z_true"][:n_samples]
            x_data[:, i] = f["x_data"][:n_samples]
            z_gen[:, i] = f["z_gen"][:n_samples]
            x_sim[:, i] = f["x_sim"][:n_samples]

        # Standardize using MC gen-level statistics only
        mu: np.floating = np.mean(z_gen[:, i])
        sigma: np.floating = np.std(z_gen[:, i])

        z_true[:, i] = (z_true[:, i] - mu) / sigma
        x_data[:, i] = (x_data[:, i] - mu) / sigma
        z_gen[:, i] = (z_gen[:, i] - mu) / sigma
        x_sim[:, i] = (x_sim[:, i] - mu) / sigma

    # Combine into dataset format: data (y=1) + MC (y=0)
    z: npt.NDArray[np.float64] = np.concatenate([z_true, z_gen], axis=0)
    x: npt.NDArray[np.float64] = np.concatenate([x_data, x_sim], axis=0)
    y: npt.NDArray[np.ubyte] = np.concatenate([
        np.ones(n_samples, dtype=np.ubyte),
        np.zeros(n_samples, dtype=np.ubyte),
    ])

    ds = RAN_Dataset(batch_size=batch_size)
    ds.dataset = ds._build_dataset(z, x, y)
    ds.splits = ds._split_dataset(ds.dataset)
    return ds.splits, n_features
