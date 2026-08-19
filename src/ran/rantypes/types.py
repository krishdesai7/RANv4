from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

# `Variables` is used as a runtime annotation by the `@jaxtyped(beartype)`
# functions in `ran.train`, and beartype has to evaluate the alias to check
# it -- so `JaxArray` cannot hide under TYPE_CHECKING. Importing `jax` here is
# safe: `ran.rantypes.types` cannot be imported without `ran.__init__` running
# first, so JAX_ENABLE_X64 is already set, and jax does not fix the Keras
# backend that `ran.baselines.omnifold` pins to tensorflow.
from jax import Array as JaxArray

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    from jax.typing import ArrayLike as JaxArrayLike
    from numpy.typing import ArrayLike, NDArray

    from ..train import TrainState

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


# `jax.value_and_grad(..., has_aux=True)` reveals as `(...) -> tuple[Any, Any]`,
# so naming the real shape here is a legal narrowing -- and it is what makes the
# destructuring at the call sites typed instead of `Any`.
type GradsAndAux = tuple[tuple[JaxArray, Variables], Variables]


class DiscGradFn(Protocol):
    def __call__(
        self,
        d_trainable: Variables,
        d_non_trainable: Variables,
        x: JaxArrayLike,
        y: JaxArrayLike,
        w: JaxArrayLike,
        /,
    ) -> GradsAndAux: ...


class GenGradFn(Protocol):
    def __call__(
        self,
        g_trainable: Variables,
        g_non_trainable: Variables,
        d_trainable: Variables,
        d_non_trainable: Variables,
        z: JaxArrayLike,
        x: JaxArrayLike,
        y: JaxArrayLike,
        /,
    ) -> GradsAndAux: ...


class TrainStep(Protocol):
    """A jitted ``(state, z, x, y) -> (state, loss)`` update."""

    def __call__(
        self,
        state: TrainState,
        z: JaxArrayLike,
        x: JaxArrayLike,
        y: JaxArrayLike,
        /,
    ) -> tuple[TrainState, JaxArray]: ...


class EvalStep(Protocol):
    """A jitted ``(state, z, x, y) -> loss`` forward pass, no updates."""

    def __call__(
        self,
        state: TrainState,
        z: JaxArrayLike,
        x: JaxArrayLike,
        y: JaxArrayLike,
        /,
    ) -> JaxArray: ...


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
