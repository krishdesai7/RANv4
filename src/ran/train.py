"""The adversarial training loop, as a single fused XLA program.

The two-optimizer min-max game does not fit ``Model.fit``, so this is hand-rolled
--- but it is not a Python loop over batches. The dataset is moved to device once
(:mod:`ran.device`), one epoch is a ``lax.scan`` over grouped batch indices, and
the epoch loop with its early stopping is a ``lax.while_loop``, so a whole run
compiles to one program and the batch gathers fuse into the first ``Dense``.

Model state lives in JAX pytrees (:class:`TrainState`) for the duration, updates
go through ``stateless_call``/``stateless_apply``, and the values are written
back into the Keras models at the end --- so the returned objects are ordinary
saveable ``keras.Model``s.

``train(fused=False)`` runs the very same epoch function from an ordinary Python
``while``. It is still one XLA program per epoch, but it keeps breakpoints,
readable tracebacks and host-side logging. Reach for it when a run goes wrong.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import keras
import numpy as np
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped

from .device import DeviceSplits, gather, train_indices
from .models import build_discriminator, build_generator

# `Variables` annotates the `@jaxtyped(beartype)` closures below, and beartype
# resolves those strings at decoration time -- so it cannot hide under
# TYPE_CHECKING.
from .rantypes import Variables  # ruff: ignore[typing-only-first-party-import]

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import PRNGKeyArray

    from .device import EvalSplit, TrainSplit
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

_HISTORY_KEYS: tuple[str, ...] = ("train_d", "train_g", "val_d", "val_g")


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


class RunCarry(NamedTuple):
    """Everything a whole run threads through ``lax.while_loop``.

    ``history`` is sized to the epoch budget up front and written row by row; the
    host slices it to ``epoch`` afterwards, since early stopping leaves a tail
    unwritten.
    """

    state: TrainState
    best_state: TrainState
    best_val: Float[Array, ""]
    wait: Int[Array, ""]
    epoch: Int[Array, ""]
    key: PRNGKeyArray
    history: Float[Array, "epochs 4"]


@jaxtyped(typechecker=beartype)
def normalize_weights(
    raw_w: Float[Array | np.ndarray, " n"],
    y: Float[Array | np.ndarray, " n"],
    mask: Float[Array | np.ndarray, " n"],
    /,
) -> Float[Array, " n"]:
    """Per-batch weights: fixed at 1 for nature, renormalized to count for MC.

    ``mask`` is 1 for a real event and 0 for a padding row, and it enters every
    sum --- so a padded eval batch gives exactly the value the unpadded one would.
    On the training path the mask is all ones and this is the plain form.
    """
    one = jnp.ones_like(y)
    n_mc = jnp.sum(mask * (one - y))
    w_mc_norm = raw_w * n_mc / (jnp.sum(mask * raw_w * (one - y)) + EPS)
    return jnp.multiply(mask, y + (one - y) * w_mc_norm)


@jaxtyped(typechecker=beartype)
def bce_sums(
    d_out: Float[Array | np.ndarray, " n"],
    y: Float[Array | np.ndarray, " n"],
    w: Float[Array | np.ndarray, " n"],
    mask: Float[Array | np.ndarray, " n"],
    /,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Masked weighted BCE, unnormalized, paired with the count it divides by.

    Handing back both halves is what lets a scan accumulate across batches and
    divide once. Reduce with ``jnp.sum(...) / n`` rather than a mean: the float64
    hazard in ``keras.ops.mean`` is gone now that this is plain ``jnp``, but the
    explicit form is what the guard test pins.
    """
    one = jnp.ones_like(d_out)
    terms = w * y * jnp.log(d_out + EPS) + w * (one - y) * jnp.log(one - d_out + EPS)
    return -jnp.sum(mask * terms), jnp.sum(mask)


@jaxtyped(typechecker=beartype)
def weighted_bce(
    d_out: Float[Array | np.ndarray, " n"],
    y: Float[Array | np.ndarray, " n"],
    w: Float[Array | np.ndarray, " n"],
    mask: Float[Array | np.ndarray, " n"],
    /,
) -> Float[Array, ""]:
    total, count = bce_sums(d_out, y, w, mask)
    return total / count


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
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
        training: bool,
    ) -> tuple[Float[Array, " n"], Variables]:
        raw_w, g_non_trainable = g.stateless_call(
            g_trainable, g_non_trainable, z, training=training
        )
        return normalize_weights(jnp.squeeze(raw_w, axis=-1), y, mask), g_non_trainable

    @jaxtyped(typechecker=beartype)
    def _disc_loss(
        d_trainable: Variables,
        d_non_trainable: Variables,
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        w: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Variables]:
        d_out, d_non_trainable = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=True,
        )
        loss = weighted_bce(jnp.squeeze(d_out, axis=-1), y, w, mask)
        return loss, d_non_trainable

    @jaxtyped(typechecker=beartype)
    def _gen_loss(
        g_trainable: Variables,
        g_non_trainable: Variables,
        d_trainable: Variables,
        d_non_trainable: Variables,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Variables]:
        w, g_non_trainable = _weights(
            g_trainable, g_non_trainable, z, y, mask, training=True
        )
        d_out, _ = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=False,
        )
        # g maximizes the BCE that d minimizes, so its loss is the negation.
        return -weighted_bce(jnp.squeeze(d_out, axis=-1), y, w, mask), g_non_trainable

    disc_grad_fn: DiscGradFn = jax.value_and_grad(fun=_disc_loss, has_aux=True)
    gen_grad_fn: GenGradFn = jax.value_and_grad(fun=_gen_loss, has_aux=True)

    @jaxtyped(typechecker=beartype)
    def disc_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[TrainState, Float[Array, ""]]:
        """One discriminator update; g is frozen."""
        # Computed outside differentiated function, so the weights are constants
        w, _ = _weights(
            state.g_trainable, state.g_non_trainable, z, y, mask, training=False
        )
        (loss, d_non_trainable), grads = disc_grad_fn(
            state.d_trainable, state.d_non_trainable, x, y, w, mask
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

    @jaxtyped(typechecker=beartype)
    def gen_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
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
            mask,
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

    @jaxtyped(typechecker=beartype)
    def eval_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Weighted BCE with no updates, both models in inference mode."""
        w, _ = _weights(
            state.g_trainable, state.g_non_trainable, z, y, mask, training=False
        )
        d_out, _ = d.stateless_call(
            trainable_variables=state.d_trainable,
            non_trainable_variables=state.d_non_trainable,
            inputs=x,
            training=False,
        )
        return bce_sums(jnp.squeeze(d_out, axis=-1), y, w, mask)

    return disc_step, gen_step, eval_step


def _make_pass(
    train: TrainSplit,
    disc_step: TrainStep,
    gen_step: TrainStep,
    *,
    batch_size: int,
    n_disc_steps: int,
):
    """Build the scanned pass over one epoch's grouped batch indices."""

    def _disc_body(
        state: TrainState, idx: Int[Array, " b"]
    ) -> tuple[TrainState, Float[Array, ""]]:
        z, x, y = gather(train, idx)
        return disc_step(state, z, x, y, jnp.ones_like(y))

    def _group(
        state: TrainState, group_idx: Int[Array, "s b"]
    ) -> tuple[TrainState, tuple[Float[Array, " s"], Float[Array, ""]]]:
        state, d_losses = lax.scan(_disc_body, state, group_idx)
        # The generator updates once per group, on the group's first batch --
        # what the host loop used to write as `step % n_disc_steps == 0`.
        z, x, y = gather(train, group_idx[0])
        state, g_loss = gen_step(state, z, x, y, jnp.ones_like(y))
        return state, (d_losses, -g_loss)

    def one_pass(
        state: TrainState, key: PRNGKeyArray
    ) -> tuple[TrainState, Float[Array, ""], Float[Array, ""]]:
        idx = train_indices(key, train.n, batch_size, n_disc_steps)
        state, (d_losses, g_losses) = lax.scan(_group, state, idx)
        return (
            state,
            jnp.sum(d_losses) / d_losses.size,
            jnp.sum(g_losses) / g_losses.size,
        )

    return one_pass


def _make_eval(eval_step: EvalStep):
    """Build the scanned forward pass over a pre-batched, masked split."""

    def evaluate(state: TrainState, split: EvalSplit) -> Float[Array, ""]:
        def _body(
            acc: tuple[Float[Array, ""], Float[Array, ""]],
            batch: tuple[Float[Array, "b d"], ...],
        ) -> tuple[tuple[Float[Array, ""], Float[Array, ""]], None]:
            z, x, y, mask = batch
            total, count = eval_step(state, z, x, y, mask)
            return (acc[0] + total, acc[1] + count), None

        zero = jnp.zeros((), dtype=split.z.dtype)
        (total, count), _ = lax.scan(
            _body, (zero, zero), (split.z, split.x, split.y, split.mask)
        )
        return total / count

    return evaluate


def _make_epoch(
    data: DeviceSplits,
    one_pass,
    evaluate,
    *,
    n_epochs: int,
    patience: int,
    min_delta: float,
):
    """Build the pure ``RunCarry -> RunCarry`` epoch, the unit of fusion."""

    def _log(
        epoch: Int[Array, ""],
        train_d: Float[Array, ""],
        train_g: Float[Array, ""],
        val_d: Float[Array, ""],
        wait: Int[Array, ""],
    ) -> None:
        logger.info(
            "Epoch %3d/%d  D: %.4f  G: %.4f  | Val D: %.4f  G: %.4f  (patience %d/%d)",
            int(epoch) + 1,
            n_epochs,
            float(train_d),
            float(train_g),
            float(val_d),
            float(val_d),
            int(wait),
            patience,
        )

    def epoch(carry: RunCarry) -> RunCarry:
        key, subkey = jax.random.split(carry.key)
        state, train_d, train_g = one_pass(carry.state, subkey)
        val_d = evaluate(state, data.val)

        improved = val_d > carry.best_val + min_delta
        best_state = cast(
            "TrainState",
            jax.tree.map(
                lambda best, cur: jnp.where(improved, cur, best),
                carry.best_state,
                state,
            ),
        )
        # d and g are scored by the same number: g's loss is the negation of the
        # BCE d minimizes, so one forward pass answers for both.
        row = jnp.stack([train_d, train_g, val_d, val_d])
        jax.debug.callback(
            _log,
            carry.epoch,
            train_d,
            train_g,
            val_d,
            jnp.where(improved, 0, carry.wait + 1),
            ordered=True,
        )
        return RunCarry(
            state=state,
            best_state=best_state,
            best_val=jnp.where(improved, val_d, carry.best_val),
            wait=jnp.where(improved, 0, carry.wait + 1),
            epoch=carry.epoch + 1,
            key=key,
            history=carry.history.at[carry.epoch].set(row),
        )

    def still_running(carry: RunCarry) -> Array:
        return (carry.epoch < n_epochs) & (carry.wait < patience)

    return epoch, still_running


def _assign(variables: list[KerasVariable], values: Variables) -> None:
    """Write JAX arrays back into a model's `keras.Variable`s."""
    for var, val in zip(variables, values, strict=False):
        var.assign(value=val)


def _initial_state(
    g: RANModel, d: RANModel, opt_g: StatelessOptimizer, opt_d: StatelessOptimizer
) -> TrainState:
    return TrainState(
        g_trainable=[v.value for v in g.trainable_variables],
        g_non_trainable=[v.value for v in g.non_trainable_variables],
        d_trainable=[v.value for v in d.trainable_variables],
        d_non_trainable=[v.value for v in d.non_trainable_variables],
        opt_g=[v.value for v in opt_g.variables],
        opt_d=[v.value for v in opt_d.variables],
    )


def _run(carry: RunCarry, epoch, still_running, *, fused: bool) -> RunCarry:
    """Drive the epoch loop, either inside XLA or from Python."""
    if fused:
        return cast(
            "RunCarry",
            jax.jit(lambda c: lax.while_loop(still_running, epoch, c))(carry),
        )
    step = jax.jit(epoch)
    while bool(still_running(carry)):
        carry = step(carry)
    return carry


def _unpack_history(history: Float[Array, "epochs 4"]) -> dict[str, list[float]]:
    rows: np.ndarray = np.asarray(history)
    return {key: rows[:, i].tolist() for i, key in enumerate(_HISTORY_KEYS)}


def train(
    splits: DatasetSplits,
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
    *,
    fused: bool = True,
) -> TrainResult:
    if seed is None:
        # No-argument SeedSequence always fills `entropy` with int drawn from OS.
        seed = cast(typ=int, val=np.random.SeedSequence().entropy) % 2**31
    keras.utils.set_random_seed(seed)

    batch_size: int = splits.train.batch_size
    data: DeviceSplits = DeviceSplits.from_splits(splits)

    g: RANModel = build_generator(dim=dim, hidden_units=hidden_units, n_layers=n_layers)
    d: RANModel = build_discriminator(
        dim=dim, hidden_units=hidden_units, n_layers=n_layers
    )
    opt_g: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_d)
    opt_g.build(g.trainable_variables)
    opt_d.build(d.trainable_variables)

    state: TrainState = _initial_state(g, d, opt_g, opt_d)
    disc_step, gen_step, eval_step = _make_steps(g, d, opt_g, opt_d)
    one_pass = _make_pass(
        data.train,
        disc_step,
        gen_step,
        batch_size=batch_size,
        n_disc_steps=n_disc_steps,
    )
    evaluate = _make_eval(eval_step)
    epoch, still_running = _make_epoch(
        data,
        one_pass,
        evaluate,
        n_epochs=n_epochs,
        patience=patience,
        min_delta=min_delta,
    )

    dtype = data.train.z.dtype
    # Batch order follows `data_seed`, not the init seed: an ensemble loop over
    # `--seed` must see identical data on every arm.
    carry = RunCarry(
        state=state,
        best_state=state,
        best_val=jnp.array(-jnp.inf, dtype=dtype),
        wait=jnp.array(0, dtype=jnp.int32),
        epoch=jnp.array(0, dtype=jnp.int32),
        key=jax.random.key(data.data_seed),
        history=jnp.zeros((n_epochs, len(_HISTORY_KEYS)), dtype=dtype),
    )

    final: RunCarry = _run(carry, epoch, still_running, fused=fused)
    n_run: int = int(final.epoch)
    if n_run < n_epochs:
        logger.info("Early stopping at epoch %d", n_run)

    # The best state, always -- not just on the early-stopping branch.
    best: TrainState = final.best_state
    _assign(g.trainable_variables, values=best.g_trainable)
    _assign(g.non_trainable_variables, values=best.g_non_trainable)
    _assign(d.trainable_variables, values=best.d_trainable)
    _assign(d.non_trainable_variables, values=best.d_non_trainable)

    test: float = float(jax.jit(evaluate)(best, data.test))
    logger.info("Test  D: %.4f  G: %.4f  (init seed %d)", test, test, seed)

    return TrainResult(g, d, _unpack_history(final.history[:n_run]), seed)
