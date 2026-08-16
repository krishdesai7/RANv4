from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..rantypes import (
    CACHE_DIR,
    CACHE_FILENAMES,
    SUBSTRUCTURE_VARIABLES,
    ZXY,
    Events,
    Populations,
)
from .datasets import DatasetSplits, RANDataset

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from numpy.typing import NDArray

logger: Logger = logging.getLogger(name=__name__)


def load_jet_dataset(
    n_samples: int = 500_000,
    batch_size: int = 1024,
    cache_dir: Path = CACHE_DIR,
    variables: frozenset[str] = SUBSTRUCTURE_VARIABLES,
    seed: int = 42,
) -> tuple[DatasetSplits, int, dict[str, tuple[np.double, np.double]]]:
    # Check cache, download if needed
    missing: list[str] = [
        v for v in variables if not (cache_dir / f"{CACHE_FILENAMES[v]}.npz").exists()
    ]
    if missing:
        from ran.data.download import download_jet_data

        logger.info(msg="Jet data not found in cache; downloading from Zenodo.")
        download_jet_data(cache_dir)

    n_features: int = len(variables)

    # Check available samples
    with np.load(file=cache_dir / f"{CACHE_FILENAMES[next(iter(variables))]}.npz") as f:
        n_avail: int = min(len(f["z_true"]), len(f["z_gen"]))
    if n_samples > n_avail:
        raise ValueError(f"Requested {n_samples} samples but only {n_avail} available")

    # Initialize arrays
    z_true: NDArray[np.double] = np.empty(
        shape=(n_samples, n_features), dtype=np.double
    )
    x_data: NDArray[np.double] = np.empty(
        shape=(n_samples, n_features), dtype=np.double
    )
    z_gen: NDArray[np.double] = np.empty(shape=(n_samples, n_features), dtype=np.double)
    x_sim: NDArray[np.double] = np.empty(shape=(n_samples, n_features), dtype=np.double)

    # Load, subsample, and standardize each variable
    std_params: dict[str, tuple[np.double, np.double]] = {}
    for i, var in enumerate(iterable=variables):
        with np.load(file=cache_dir / f"{CACHE_FILENAMES[var]}.npz") as f:
            z_true[:, i] = f["z_true"][:n_samples]
            x_data[:, i] = f["x_data"][:n_samples]
            z_gen[:, i] = f["z_gen"][:n_samples]
            x_sim[:, i] = f["x_sim"][:n_samples]

        # Standardize using MC gen-level statistics only
        mu: np.double = z_gen[:, i].mean(dtype=np.double)
        sigma: np.double = z_gen[:, i].std(dtype=np.double)
        std_params[var] = (mu, sigma)

        z_true[:, i] = (z_true[:, i] - mu) / sigma
        x_data[:, i] = (x_data[:, i] - mu) / sigma
        z_gen[:, i] = (z_gen[:, i] - mu) / sigma
        x_sim[:, i] = (x_sim[:, i] - mu) / sigma

    data: ZXY[np.double] = Populations(
        mc=Events(z_gen, x_sim), data=x_data, truth=z_true
    ).interleave()

    splits: DatasetSplits[np.double] = RANDataset(
        batch_size=batch_size, seed=seed
    ).splits_from_data(data)
    return splits, n_features, std_params
