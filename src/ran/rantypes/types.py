from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from jax._src.basearray import Array as JaxArray
    from numpy.typing import NDArray

# ---------------------------------
# Plotting
# ---------------------------------


class VarInfo(TypedDict):
    xlim: tuple[float, float]
    xlabel: str
    symbol: str
    mu: float
    sigma: float


# ---------------------------------
# Training
# ---------------------------------
type Variables = list[JaxArray]


# ---------------------------------
# Baselines
# ---------------------------------


class MetricRecord(TypedDict):
    """Before/after scores for one variable at one level."""

    wasserstein_before: float
    wasserstein_after: float
    wasserstein_improvement_pct: float
    jensenshannon_before: float
    jensenshannon_after: float
    jensenshannon_improvement_pct: float
    triangular_before: float
    triangular_after: float
    triangular_improvement_pct: float


# ---------------------------------
# Data
# ---------------------------------
type Nested[T] = T | list[Nested[T]]

type Batch = tuple[dict[str, NDArray[np.double]], NDArray[np.ubyte]]
