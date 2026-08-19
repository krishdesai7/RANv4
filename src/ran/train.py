from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import keras
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Real, jaxtyped
from keras import ops

from .models import build_discriminator, build_generator
from .rantypes import ZXY, Events, Variables

if TYPE_CHECKING:
    from logging import Logger

    from numpy.typing import NDArray

    from .data import ArrayDataset
    from .rantypes import (
        DatasetSplits,
        DiscGradFn,
        EvalStep,
        GenGradFn,
        KerasVariable,
        RANModel,
        StatelessOptimizer,
        TrainStep,
    )

logger: Logger = logging.getLogger(name=__name__)

if keras.backend.backend() != "jax":
    # Importing `keras` before `ran` wins the race for the backend, and the
    # jitted steps below fail deep inside a trace.
    raise RuntimeError(
        f"ran.train requires the JAX backend, got {keras.backend.backend()!r}. "
        "Import `ran` (or any ran.* module) before `keras`, or set "
        "KERAS_BACKEND=jax in the environment."
    )

EPS: float = keras.config.epsilon()


class TrainResult(NamedTuple):
    """Return package of a model training. Unpacks as ``(g, d, history, seed)``."""

    g: RANModel
    d: RANModel
    history: dict[str, list[float]]
    seed: int


class TrainState(NamedTuple):
    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables
    opt_g: Variables
    opt_d: Variables


@jaxtyped(typechecker=beartype)
def normalize_weights(
    raw_w: Float[Array | np.ndarray, " n"],
    y: Real[Array | np.ndarray, " n"],
    /,
) -> Float[Array, " n"]:
    one = ops.ones_like(y)
    n_mc = ops.sum(one - y)
    w_mc_norm = raw_w * n_mc / (ops.sum(raw_w * (one - y)) + EPS)
    return y + (one - y) * w_mc_norm


@jaxtyped(typechecker=beartype)
def weighted_bce(
    d_out: Float[Array | np.ndarray, " n"],
    y: Real[Array | np.ndarray, " n"],
    w: Float[Array | np.ndarray, " n"],
    /,
) -> Float[Array, ""]:
    one = ops.ones_like(d_out)
    terms = w * y * ops.log(d_out + EPS) + w * (one - y) * ops.log(one - d_out + EPS)
    return -ops.sum(terms) / ops.shape(terms)[0]


def _make_steps(
    g: RANModel,
    d: RANModel,
    opt_g: StatelessOptimizer,
    opt_d: StatelessOptimizer,
) -> tuple[TrainStep, TrainStep, EvalStep]:
    @jaxtyped(typechecker=beartype)
    def _weights(
        g_trainable: Variables,
        g_non_trainable: Variables,
        z: Float[Array, " n d"],
        y: Real[Array, " n"],
        training: bool,
    ) -> tuple[Float[Array, " n"], Variables]:
        raw_w, g_non_trainable = g.stateless_call(
            g_trainable, g_non_trainable, z, training=training
        )
        return normalize_weights(ops.squeeze(raw_w, axis=-1), y), g_non_trainable

    @jaxtyped(typechecker=beartype)
    def _disc_loss(
        d_trainable: Variables,
        d_non_trainable: Variables,
        x: Float[Array, " n d"],
        y: Real[Array, " n"],
        w: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Variables]:
        d_out, d_non_trainable = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=True,
        )
        return weighted_bce(ops.squeeze(d_out, axis=-1), y, w), d_non_trainable

    @jaxtyped(typechecker=beartype)
    def _gen_loss(
        g_trainable: Variables,
        g_non_trainable: Variables,
        d_trainable: Variables,
        d_non_trainable: Variables,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Real[Array, " n"],
    ) -> tuple[Float[Array, ""], Variables]:
        w, g_non_trainable = _weights(g_trainable, g_non_trainable, z, y, training=True)
        d_out, _ = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=False,
        )
        # g maximizes the BCE that d minimizes, so its loss is the negation.
        return -weighted_bce(ops.squeeze(d_out, axis=-1), y, w), g_non_trainable

    disc_grad_fn: DiscGradFn = jax.value_and_grad(fun=_disc_loss, has_aux=True)
    gen_grad_fn: GenGradFn = jax.value_and_grad(fun=_gen_loss, has_aux=True)

    @jax.jit
    @jaxtyped(typechecker=beartype)
    def disc_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Real[Array, " n"],
    ) -> tuple[TrainState, Float[Array, ""]]:
        """One discriminator update; g is frozen."""
        # Computed outside differentiated function, so the weights are constants
        w, _ = _weights(state.g_trainable, state.g_non_trainable, z, y, training=False)
        (loss, d_non_trainable), grads = disc_grad_fn(
            state.d_trainable, state.d_non_trainable, x, y, w
        )
        d_trainable, opt_d_vars = opt_d.stateless_apply(
            optimizer_variables=state.opt_d,
            grads=grads,
            trainable_variables=state.d_trainable,
        )
        return (
            state._replace(
                d_trainable=d_trainable,
                d_non_trainable=d_non_trainable,
                opt_d=opt_d_vars,
            ),
            loss,
        )

    @jax.jit
    @jaxtyped(typechecker=beartype)
    def gen_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Real[Array, " n"],
    ) -> tuple[TrainState, Float[Array, ""]]:
        """One generator update; d is frozen (it enters only as a constant)."""
        (loss, g_non_trainable), grads = gen_grad_fn(
            state.g_trainable,
            state.g_non_trainable,
            state.d_trainable,
            state.d_non_trainable,
            z,
            x,
            y,
        )
        g_trainable, opt_g_vars = opt_g.stateless_apply(
            optimizer_variables=state.opt_g,
            grads=grads,
            trainable_variables=state.g_trainable,
        )
        return (
            state._replace(
                g_trainable=g_trainable,
                g_non_trainable=g_non_trainable,
                opt_g=opt_g_vars,
            ),
            loss,
        )

    @jax.jit
    @jaxtyped(typechecker=beartype)
    def eval_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Real[Array, " n"],
    ) -> Float[Array, ""]:
        """Weighted BCE with no updates, both models in inference mode."""
        w, _ = _weights(state.g_trainable, state.g_non_trainable, z, y, training=False)
        d_out, _ = d.stateless_call(
            trainable_variables=state.d_trainable,
            non_trainable_variables=state.d_non_trainable,
            inputs=x,
            training=False,
        )
        return weighted_bce(ops.squeeze(d_out, axis=-1), y, w)

    return disc_step, gen_step, eval_step


def _as_batch[T: np.floating = np.double](
    features: dict[str, NDArray[T]], y: NDArray[np.ubyte]
) -> ZXY[T]:
    return ZXY(
        Events(
            z=features["z"],
            x=features["x"],
        ),
        y=y.reshape(-1),
    )


def _run_epoch[T: np.floating = np.double](
    state: TrainState,
    train_ds: ArrayDataset[T],
    disc_step: TrainStep,
    gen_step: TrainStep,
    n_disc_steps: int,
) -> tuple[TrainState, T, T]:
    n_batches: int = len(train_ds)
    d_losses: NDArray[T] = np.empty(shape=n_batches, dtype=train_ds.dtype)
    # g updates on every n_disc_steps-th batch, so its curve has fewer points.
    # Sized to exactly those, or the unwritten slots would enter the mean.
    g_losses: NDArray[T] = np.empty(
        shape=math.ceil(n_batches / n_disc_steps), dtype=train_ds.dtype
    )
    for step, (features, y) in enumerate(train_ds):
        zxy: ZXY[T] = _as_batch(features, y)

        state, d_loss = disc_step(state, zxy.z, zxy.x, zxy.y)
        d_losses[step] = d_loss

        if step % n_disc_steps == 0:
            state, g_loss = gen_step(state, zxy.z, zxy.x, zxy.y)
            g_losses[step // n_disc_steps] = -g_loss

    return state, d_losses.mean(), g_losses.mean()


def _eval_dataset[T: np.floating = np.double](
    eval_step: EvalStep, state: TrainState, dataset: ArrayDataset[T]
) -> tuple[T, T]:
    """Mean weighted BCE over a split, as (d_loss, g_loss)."""
    total: float = 0.0
    for features, y in dataset:
        zxy: ZXY[T] = _as_batch(features, y)
        total += float(eval_step(state, zxy.z, zxy.x, zxy.y))
    mean: T = np.divide(total, len(dataset), dtype=dataset.dtype)
    return mean, mean


def _assign(variables: list[KerasVariable], values: Variables) -> None:
    """Write JAX arrays back into a model's `keras.Variable`s."""
    for var, val in zip(variables, values, strict=False):
        var.assign(value=val)


def train[T: np.floating = np.double](
    splits: DatasetSplits[T],
    dim: int,
    hidden_units: int,
    n_layers: int,
    seed: int | None,
    patience: int,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    min_delta: float = 0.0001,
) -> TrainResult:
    if seed is None:
        # No-argument SeedSequence always fills `entropy` with int drawn from OS.
        seed = cast(typ=int, val=np.random.SeedSequence().entropy) % 2**31
    keras.utils.set_random_seed(seed)
    # Rewind the batch-order sequence so repeated runs over one DatasetSplits
    # i.e., an ensemble loop over init seeds, all see identical data.
    splits.train.reset()

    g: RANModel = build_generator(dim=dim, hidden_units=hidden_units, n_layers=n_layers)
    d: RANModel = build_discriminator(
        dim=dim, hidden_units=hidden_units, n_layers=n_layers
    )
    opt_g: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_d)
    opt_g.build(g.trainable_variables)
    opt_d.build(d.trainable_variables)

    state = TrainState(
        g_trainable=[v.value for v in g.trainable_variables],
        g_non_trainable=[v.value for v in g.non_trainable_variables],
        d_trainable=[v.value for v in d.trainable_variables],
        d_non_trainable=[v.value for v in d.non_trainable_variables],
        opt_g=[v.value for v in opt_g.variables],
        opt_d=[v.value for v in opt_d.variables],
    )
    disc_step, gen_step, eval_step = _make_steps(g, d, opt_g, opt_d)

    history: dict[str, list[float]] = {
        "train_d": [],
        "train_g": [],
        "val_d": [],
        "val_g": [],
    }
    best_val_d: float = -math.inf
    best_state: TrainState | None = None
    wait: int = 0

    for epoch in range(n_epochs):
        state, mean_td, mean_tg = _run_epoch(
            state, splits.train, disc_step, gen_step, n_disc_steps
        )
        mean_val: tuple[T, T] = _eval_dataset(eval_step, state, dataset=splits.val)

        history["train_d"].append(float(mean_td))
        history["train_g"].append(float(mean_tg))
        history["val_d"].append(float(mean_val[0]))
        history["val_g"].append(float(mean_val[1]))

        # Early stopping: higher val D = better convergence toward log(2)
        if mean_val[0] > best_val_d + min_delta:
            best_val_d = float(mean_val[0])
            best_state = state
            wait = 0
        else:
            wait += 1

        logger.info(
            "Epoch %3d/%d  D: %.4f  G: %.4f  | Val D: %.4f  G: %.4f  (patience %d/%d)",
            epoch + 1,
            n_epochs,
            mean_td,
            mean_tg,
            mean_val[0],
            mean_val[1],
            wait,
            patience,
        )

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            if best_state is not None:
                state: TrainState = best_state
            break

    _assign(g.trainable_variables, values=state.g_trainable)
    _assign(g.non_trainable_variables, values=state.g_non_trainable)
    _assign(d.trainable_variables, values=state.d_trainable)
    _assign(d.non_trainable_variables, values=state.d_non_trainable)

    # Final test evaluation
    test: tuple[T, T] = _eval_dataset(eval_step, state, dataset=splits.test)
    logger.info("Test  D: %.4f  G: %.4f  (init seed %d)", test[0], test[1], seed)

    return TrainResult(g, d, history, seed)
