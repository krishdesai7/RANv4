from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy.linalg import cholesky

from ..rantypes import REQUIRED_KEYS, GaussianConfig

if TYPE_CHECKING:
    from collections.abc import Mapping
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


def gaussian_config_from_run_config(
    params: Mapping[str, Any], dim: int
) -> GaussianConfig:
    """Rebuild a `GaussianConfig` from the `gaussian_params` block of a run's
    config.json, across every format runs/ has ever held.

    Three of them exist on disk, and two share their key names:

    ``cov_gen``            covariance matrices, written since the type refactor.
    ``sigma_gen`` (list)   covariance matrices under the old name -- master's
                           `__main__` stored `cov_gen` as `sigma_gen`.
    ``sigma_gen`` (scalar) a raw sigma, from before that, needing promotion.

    Routing the ``sigma_*`` spelling through `sigma_to_covariance` resolves the
    ambiguity without guessing: a raw scalar promotes to σ²I, a vector to
    diag(σ²), and an already-formed matrix passes through unchanged (checked for
    shape, symmetry and positive-definiteness on the way). So both readings land
    on the same covariance, and the ``cov_*`` spelling needs no promotion at all.
    """
    missing = {"mu_gen", "mu_true"} - params.keys()
    if missing:
        raise ValueError(f"gaussian_params missing required keys: {missing}")

    def _covariance(name: str) -> NDArray[np.double]:
        if f"cov_{name}" in params:
            return _full_covariance(
                np.asarray(params[f"cov_{name}"], dtype=np.double), dim
            )
        if f"sigma_{name}" in params:
            return sigma_to_covariance(params[f"sigma_{name}"], dim)
        raise ValueError(f"gaussian_params has neither cov_{name} nor sigma_{name}")

    return GaussianConfig(
        dim,
        np.asarray(params["mu_gen"], dtype=np.double).ravel(),
        np.asarray(params["mu_true"], dtype=np.double).ravel(),
        _covariance("gen"),
        _covariance("true"),
        _covariance("detector"),
    )


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
