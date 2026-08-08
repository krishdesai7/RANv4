from __future__ import annotations

from typing import TYPE_CHECKING, cast

import keras

if TYPE_CHECKING:
    from typing import Protocol

    from .rantypes import RANModel

    class _InputFactory(Protocol):
        def __call__(
            self, *, shape: tuple[int | None, ...], dtype: str
        ) -> keras.KerasTensor: ...

    class _DenseLayer(Protocol):
        def __call__(self, inputs: keras.KerasTensor) -> keras.KerasTensor: ...

    class _DenseFactory(Protocol):
        def __call__(
            self, units: int, *, activation: str, dtype: str
        ) -> _DenseLayer: ...

    class _ModelFactory(Protocol):
        def __call__(
            self,
            inputs: keras.KerasTensor,
            outputs: keras.KerasTensor,
            *,
            name: str,
        ) -> RANModel: ...


_keras_input = cast("_InputFactory", keras.Input)
_keras_dense = cast("_DenseFactory", keras.layers.Dense)
_keras_model = cast("_ModelFactory", keras.Model)


def build_generator(
    dim: int = 1, hidden_units: int = 64, n_layers: int = 2
) -> RANModel:
    """g(z): nominal-level events -> per-event weights."""
    inputs = _keras_input(shape=(dim,), dtype="float64")
    x = inputs
    for _ in range(n_layers):
        x = _keras_dense(hidden_units, activation="relu", dtype="float64")(x)
    x = _keras_dense(1, activation="softplus", dtype="float64")(x)
    return _keras_model(inputs, x, name="generator")


def build_discriminator(
    dim: int = 1, hidden_units: int = 64, n_layers: int = 2
) -> RANModel:
    """d(x): reco-level events -> data vs MC probability."""
    inputs = _keras_input(shape=(dim,), dtype="float64")
    x = inputs
    for _ in range(n_layers):
        x = _keras_dense(hidden_units, activation="relu", dtype="float64")(x)
    x = _keras_dense(1, activation="sigmoid", dtype="float64")(x)
    return _keras_model(inputs, x, name="discriminator")
