"""Validated views of the two things a run is configured by: a Gaussian
parameter set and a run's own `config.json`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any, Final, Literal

    import numpy as np
    from numpy.typing import NDArray


REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    (
        "mu_gen",
        "mu_true",
        "sigma_gen",
        "sigma_true",
        "sigma_detector",
    )
)


class GaussianConfig(NamedTuple):
    dim: int
    mu_gen: NDArray[np.double]
    mu_true: NDArray[np.double]
    cov_gen: NDArray[np.double]
    cov_true: NDArray[np.double]
    cov_detector: NDArray[np.double]

    def model_dump(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "mu_gen": self.mu_gen.tolist(),
            "mu_true": self.mu_true.tolist(),
            "cov_gen": self.cov_gen.tolist(),
            "cov_true": self.cov_true.tolist(),
            "cov_detector": self.cov_detector.tolist(),
        }


@dataclass(frozen=True)
class RunConfig:
    """A validated view of a run's config.json.

    `source` is the raw dict, kept because `_load_splits` reconstructs the
    dataset from it and must see exactly what the run recorded.
    """

    source: dict[str, Any]
    dataset: Literal["gaussian", "jets"]
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]
