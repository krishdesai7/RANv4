from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy.linalg import cholesky

from ..rantypes import REQUIRED_KEYS, GaussianConfig

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


def _scalar_covariance(arr: NDArray[np.double], dim: int) -> NDArray[np.double]:
    """(σ²)I from a single sigma."""
    val: np.double = arr.ravel()[0]
    if val < 0:
        raise ValueError(f"sigma scalar must be non-negative, got {val}")
    return val**2 * np.identity(dim, dtype=np.double)


def _diagonal_covariance(arr: NDArray[np.double], dim: int) -> NDArray[np.double]:
    """diag(σ²) from a per-dimension sigma vector."""
    if arr.shape[0] != dim:
        raise ValueError(f"sigma vector has length {arr.shape[0]}, expected {dim = }")
    if np.any(arr < 0):
        raise ValueError("sigma vector elements must be non-negative")
    return np.diag(arr**2).astype(np.double)


def _full_covariance(arr: NDArray[np.double], dim: int) -> NDArray[np.double]:
    """An already-formed covariance matrix, checked for shape and symmetry."""
    if arr.shape != (dim, dim):
        raise ValueError(f"sigma matrix has shape {arr.shape}, expected {dim = }")
    if not np.allclose(arr, arr.T):
        raise ValueError("sigma matrix must be symmetric")
    return arr


def sigma_to_covariance(
    sigma: ArrayLike,
    dim: int,
) -> NDArray[np.double]:
    arr: NDArray[np.double] = np.atleast_1d(np.asarray(sigma, dtype=np.double))

    cov: NDArray[np.double]
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        cov = _scalar_covariance(arr, dim)
    elif arr.ndim == 1:
        cov = _diagonal_covariance(arr, dim)
    elif arr.ndim == 2:
        cov = _full_covariance(arr, dim)
    else:
        raise ValueError(f"sigma must be 0D, 1D, or 2D, got {arr.ndim = }")

    cholesky(cov, lower=True)
    return cov


def parse_gaussian_config(config_path: Path) -> GaussianConfig:
    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    missing: frozenset[str] = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    mu_gen: NDArray[np.double] = np.asarray(raw["mu_gen"], dtype=np.double).ravel()
    mu_true: NDArray[np.double] = np.asarray(raw["mu_true"], dtype=np.double).ravel()

    dim: int = mu_gen.shape[0]
    if mu_true.shape[0] != dim:
        raise ValueError(f"mu_true has dim {mu_true.shape[0]}, mu_gen has {dim=}")

    cov_gen: NDArray[np.double] = sigma_to_covariance(raw["sigma_gen"], dim)
    cov_true: NDArray[np.double] = sigma_to_covariance(raw["sigma_true"], dim)
    cov_detector: NDArray[np.double] = sigma_to_covariance(raw["sigma_detector"], dim)

    return GaussianConfig(
        dim,
        mu_gen,
        mu_true,
        cov_gen,
        cov_true,
        cov_detector,
    )
