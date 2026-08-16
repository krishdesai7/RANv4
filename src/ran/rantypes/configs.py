from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any, Final, Literal, LiteralString

    import numpy as np
    from numpy.typing import NDArray


REQUIRED_KEYS: Final[frozenset[LiteralString]] = frozenset(
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
    source: dict[str, Any]
    dataset: Literal["gaussian", "jets"]
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]
