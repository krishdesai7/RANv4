from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

from jax._src.basearray import Array as JaxArray
from ran.rantypes import KerasVariable, RANModel, StatelessOptimizer, Variables

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


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
