from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

import yaml


def sigma_to_covariance(
    sigma: float | list | npt.NDArray,
    dim: int,
) -> npt.NDArray[np.double]:
    """Promote sigma (scalar, vector, or matrix)
    to a (dim, dim) covariance matrix,
    where dim is the dimension of the data.

    - scalar: σ²I
    - (dim,) vector: diag(σ²)
    - (dim, dim) matrix: used as-is

    Validates positive-definiteness via Cholesky decomposition.
    """
    arr: npt.NDArray[np.double] = np.atleast_1d(np.asarray(sigma,
                                                dtype=np.double))

    cov: npt.NDArray[np.double]
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        val = float(arr.ravel()[0])
        if val < 0:
            raise ValueError(f"sigma scalar must be non-negative, got {val}")
        cov = val ** 2 * np.eye(dim, dtype=np.double)
    elif arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(
                f"sigma vector has length {arr.shape[0]}, expected dim={dim}"
            )
        if np.any(arr < 0):
            raise ValueError("sigma vector elements must be non-negative")
        cov = np.diag(arr ** 2).astype(np.double)
    elif arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(
                f"sigma matrix has shape {arr.shape}, expected dim={dim}"
            )
        if not np.allclose(arr, arr.T):
            raise ValueError("sigma matrix must be symmetric")
        cov = arr
    else:
        raise ValueError(f"sigma must be 0D, 1D, or 2D, got ndim={arr.ndim}")

    cholesky(cov, lower=True)
    return cov


REQUIRED_KEYS: set[str] = {"mu_gen", "mu_true",
                           "sigma_gen", "sigma_true", "sigma_detector"}


def parse_gaussian_config(config_path: str | Path) -> dict:
    """Parse a Gaussian YAML config file.

    Returns a dict with keys:
        dim (int), mu_gen, mu_true (1D arrays),
        cov_gen, cov_true, cov_detector (2D covariance matrices).
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        raw: dict = yaml.safe_load(f)

    missing: set[str] = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    mu_gen: npt.NDArray[np.double] = np.asarray(raw["mu_gen"],
                                                dtype=np.double).ravel()
    mu_true: npt.NDArray[np.double] = np.asarray(raw["mu_true"],
                                                 dtype=np.double).ravel()

    dim: int = mu_gen.shape[0]
    if mu_true.shape[0] != dim:
        raise ValueError(
            f"mu_true has dim {mu_true.shape[0]}, mu_gen has {dim=}"
        )

    cov_gen: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_gen"], dim)
    cov_true: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_true"], dim)
    cov_detector: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_detector"], dim)

    return {
        "dim": dim,
        "mu_gen": mu_gen,
        "mu_true": mu_true,
        "cov_gen": cov_gen,
        "cov_true": cov_true,
        "cov_detector": cov_detector,
    }
