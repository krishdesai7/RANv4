from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    from jax._src.basearray import Array as JaxArray
    from numpy.typing import ArrayLike, NDArray

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


class KerasVariable(Protocol):
    @property
    def value(self) -> JaxArray: ...

    def assign(self, value: JaxArray) -> None: ...


class RANModel(Protocol):
    @property
    def trainable_variables(self) -> list[KerasVariable]: ...

    @property
    def non_trainable_variables(self) -> list[KerasVariable]: ...

    def __call__(self, inputs: ArrayLike) -> JaxArray: ...

    def stateless_call(
        self,
        trainable_variables: Variables,
        non_trainable_variables: Variables,
        inputs: ArrayLike,
        *,
        training: bool,
    ) -> tuple[JaxArray, Variables]: ...

    def save(self, filepath: str | PathLike[str]) -> None: ...


class StatelessOptimizer(Protocol):
    @property
    def variables(self) -> list[KerasVariable]: ...

    def build(self, variables: list[KerasVariable]) -> None: ...

    def stateless_apply(
        self,
        optimizer_variables: Variables,
        grads: Variables,
        trainable_variables: Variables,
    ) -> tuple[Variables, Variables]: ...


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

type Batch[T: np.floating = np.double] = tuple[dict[str, NDArray[T]], NDArray[np.ubyte]]
