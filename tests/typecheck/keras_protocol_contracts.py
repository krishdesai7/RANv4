from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, assert_type

from jax._src.basearray import Array as JaxArray
from ran.evaluate import _get_weights as evaluate_weights
from ran.models import build_discriminator, build_generator
from ran.plotting import _get_weights as plotting_weights
from ran.rantypes import (
    KerasVariable,
    RANModel,
    StatelessOptimizer,
    Variables,
)
from ran.train import TrainResult, _make_steps
from ran.workflow import _load_artifacts

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def check_protocol_surfaces(
    model: RANModel,
    optimizer: StatelessOptimizer,
    variable: KerasVariable,
    inputs: ArrayLike,
    state: Variables,
) -> None:
    assert_type(model.trainable_variables, list[KerasVariable])
    assert_type(model.non_trainable_variables, list[KerasVariable])
    assert_type(model(inputs), JaxArray)
    output, next_state = model.stateless_call(state, state, inputs, training=True)
    assert_type(output, JaxArray)
    assert_type(next_state, Variables)
    model.save("model.keras")

    assert_type(variable.value, JaxArray)
    variable.assign(state[0])

    optimizer.build(model.trainable_variables)
    assert_type(optimizer.variables, list[KerasVariable])
    trainable, optimizer_state = optimizer.stateless_apply(state, state, state)
    assert_type(trainable, Variables)
    assert_type(optimizer_state, Variables)


def check_training_boundary(model: RANModel, optimizer: StatelessOptimizer) -> None:
    assert_type(build_generator(), RANModel)
    assert_type(build_discriminator(), RANModel)
    _make_steps(model, model, optimizer, optimizer)


def check_model_consumers(
    model: RANModel,
    inputs: NDArray,
    result: TrainResult,
) -> None:
    evaluate_weights(model, inputs)
    plotting_weights(model, inputs)
    assert_type(result.g, RANModel)
    loaded, _ = _load_artifacts(Path("run"))
    assert_type(loaded, RANModel)
