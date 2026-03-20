"""Load jet substructure data for RAN training.

Checks .cache/ for per-variable .npz files. If missing, invokes
download_jet_data to fetch from Zenodo. Loads, subsamples, z-score
standardizes (using MC gen-level statistics only), and builds a
tf.data.Dataset via RAN_Dataset.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ran.data.datasets import RAN_Dataset, DatasetSplits
from ran.data.download import CACHE_FILENAMES

SUBSTRUCTURE_VARIABLES = ("m", "M", "w", "tau21", "zg", "sdm")
CACHE_DIR = Path(".cache")

JET_OBS: dict[str, dict] = {
    "m":     {"xlim": (0, 75),   "xlabel": "Jet Mass",                       "symbol": r"$m$ [GeV]"},
    "M":     {"xlim": (0, 80),   "xlabel": "Jet Constituent Multiplicity",   "symbol": r"$M$"},
    "w":     {"xlim": (0, 0.6),  "xlabel": "Jet Width",                      "symbol": r"$w$"},
    "tau21": {"xlim": (0, 1.2),  "xlabel": r"$N$-subjettiness Ratio",        "symbol": r"$\tau_{21}^{(\beta=1)}$"},
    "zg":    {"xlim": (0, 0.5),  "xlabel": "Groomed Jet Momentum Fraction",  "symbol": r"$z_g$"},
    "sdm":   {"xlim": (-14, -2), "xlabel": "Soft Drop Jet Mass",             "symbol": r"$\ln\rho$"},
}


def load_jet_dataset(
    n_samples: int = 500_000,
    batch_size: int = 1024,
    cache_dir: Path = CACHE_DIR,
    variables: tuple[str, ...] = SUBSTRUCTURE_VARIABLES,
) -> tuple[DatasetSplits, int, dict[str, tuple[np.double, np.double]]]:
    """Load jet substructure data and return DatasetSplits.

    Each selected substructure variable is z-score standardized using
    the MC gen-level (z_gen) mean and std. The same parameters are applied
    to all four arrays (z_true, x_data, z_gen, x_sim) to avoid information
    leakage and preserve correlations.

    Args:
        n_samples: Number of events to use per class (data and MC).
        batch_size: Batch size for tf.data.Dataset.
        cache_dir: Directory containing per-variable .npz files.
        variables: Which substructure variables to use.

    Returns:
        (splits, dim, std_params): DatasetSplits, feature dimensionality,
            and standardization parameters {var_name: (mu, sigma)}.
    """
    # Check cache, download if needed
    missing: list[str] = [v for v in variables
               if not (cache_dir / f"{CACHE_FILENAMES[v]}.npz").exists()]
    if missing:
        from ran.data.download import download_jet_data
        print("Cached jet data not found. Downloading from Zenodo...")
        download_jet_data(cache_dir)

    n_features: int = len(variables)

    # Check available samples
    with np.load(cache_dir / f"{CACHE_FILENAMES[variables[0]]}.npz") as f:
        n_avail: int = min(len(f["z_true"]), len(f["z_gen"]))
    if n_samples > n_avail:
        raise ValueError(
            f"Requested {n_samples} samples but only {n_avail} available"
        )

    # Initialize arrays
    z_true: npt.NDArray[np.double] = np.empty((n_samples, n_features), dtype=np.double)
    x_data: npt.NDArray[np.double] = np.empty((n_samples, n_features), dtype=np.double)
    z_gen: npt.NDArray[np.double] = np.empty((n_samples, n_features), dtype=np.double)
    x_sim: npt.NDArray[np.double] = np.empty((n_samples, n_features), dtype=np.double)

    # Load, subsample, and standardize each variable
    std_params: dict[str, tuple[np.double, np.double]] = {}
    for i, var in enumerate(variables):
        with np.load(cache_dir / f"{CACHE_FILENAMES[var]}.npz") as f:
            z_true[:, i] = f["z_true"][:n_samples]
            x_data[:, i] = f["x_data"][:n_samples]
            z_gen[:, i] = f["z_gen"][:n_samples]
            x_sim[:, i] = f["x_sim"][:n_samples]

        # Standardize using MC gen-level statistics only
        mu: np.double = np.mean(z_gen[:, i], dtype=np.double)
        sigma: np.double = np.std(z_gen[:, i], dtype=np.double)
        std_params[var] = (mu, sigma)

        z_true[:, i] = (z_true[:, i] - mu) / sigma
        x_data[:, i] = (x_data[:, i] - mu) / sigma
        z_gen[:, i] = (z_gen[:, i] - mu) / sigma
        x_sim[:, i] = (x_sim[:, i] - mu) / sigma

    # Combine into dataset format: nature (y=1) + MC (y=0)
    z: npt.NDArray[np.double] = np.concatenate([z_true, z_gen], axis=0)
    x: npt.NDArray[np.double] = np.concatenate([x_data, x_sim], axis=0)
    y: npt.NDArray[np.ubyte] = np.concatenate([
        np.ones(n_samples, dtype=np.ubyte),
        np.zeros(n_samples, dtype=np.ubyte),
    ])

    ds = RAN_Dataset(batch_size=batch_size)
    ds.dataset = ds._build_dataset(z, x, y)
    ds.splits = ds._split_dataset(ds.dataset)
    return ds.splits, n_features, std_params
