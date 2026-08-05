"""Adversarial training loop for RAN, on Keras 3 with the JAX backend.

The min-max game needs two optimizers driven at different cadences against a
shared loss, which does not fit `Model.fit`, so this is a hand-rolled loop. It
follows the standard Keras 3 + JAX pattern: model state lives in plain JAX
pytrees (never in the `keras.Variable`s) for the duration of training, updates
go through `stateless_call`/`stateless_apply`, and each step is a single jitted
function. Values are written back into the Keras models at the end so the
returned objects are ordinary, saveable `keras.Model`s.

The loss math below is written in backend-agnostic `keras.ops`; only the
gradient transform and jit are native JAX.
"""

import logging
from typing import NamedTuple

import jax
import keras
import numpy as np
import numpy.typing as npt
from keras import ops

from ran.data import ArrayDataset, DatasetSplits
from ran.models import build_discriminator, build_generator

logger = logging.getLogger(__name__)

if keras.backend.backend() != "jax":
    # Importing `keras` before `ran` wins the race for the backend, and the
    # jitted steps below would then fail deep inside a trace. Say so up front.
    raise RuntimeError(
        f"ran.train requires the JAX backend, got {keras.backend.backend()!r}. "
        "Import `ran` (or any ran.* module) before `keras`, or set "
        "KERAS_BACKEND=jax in the environment."
    )

EPS: float = keras.config.epsilon()

type Variables = list[jax.Array]


class TrainResult(NamedTuple):
    """What `train` returns. Unpacks as ``(g, d, history, seed)``."""

    g: keras.Model
    d: keras.Model
    history: dict[str, list[float | np.floating]]
    seed: int


class TrainState(NamedTuple):
    """All mutable training state, as a JAX pytree.

    Held outside the `keras.Model`s so jitted steps stay pure and no
    host/device sync happens between steps.
    """

    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables
    opt_g: Variables
    opt_d: Variables


def normalize_weights(raw_w, y):
    """Per-event weights: 1 for data, mean-preserving g(z) for MC.

    `raw_w` is the raw generator output for every event in the batch. Data
    events (y=1) are pinned to weight 1; MC events (y=0) are rescaled so their
    weights sum to the MC event count, preserving the per-class normalization.

    The y=1 entries of `raw_w` are multiplied by (1 - y) = 0 in both the sum and
    the result, so g's output on data rows -- which are z_true -- cannot reach
    the loss or its gradient. That is what keeps z_true out of the model.
    """
    one = ops.ones_like(y)
    n_mc = ops.sum(one - y)
    w_mc_norm = raw_w * n_mc / (ops.sum(raw_w * (one - y)) + EPS)
    return y + (one - y) * w_mc_norm


def weighted_bce(d_out, y, w):
    """Weighted binary cross-entropy.

    Reduced with `ops.sum(...) / n` rather than `ops.mean`: for float64 input
    `keras.ops.mean` picks a float32 compute dtype internally and returns a
    float64 result carrying ~1e-8 relative error, which would silently undo the
    float64 policy this project runs on. `ops.sum` accumulates in float64.
    """
    one = ops.ones_like(d_out)
    terms = w * y * ops.log(d_out + EPS) + w * (one - y) * ops.log(one - d_out + EPS)
    return -ops.sum(terms) / ops.shape(terms)[0]


def _make_steps(
    g: keras.Model,
    d: keras.Model,
    opt_g: keras.optimizers.Optimizer,
    opt_d: keras.optimizers.Optimizer,
):
    """Build the jitted disc/gen/eval steps, closing over the models.

    The models are captured rather than passed so jit sees only array
    arguments; each returned function is traced once per input shape.
    """

    def _weights(g_trainable, g_non_trainable, z, y, training: bool):
        raw_w, g_non_trainable = g.stateless_call(
            g_trainable, g_non_trainable, z, training=training
        )
        return normalize_weights(ops.squeeze(raw_w, axis=-1), y), g_non_trainable

    def _disc_loss(d_trainable, d_non_trainable, x, y, w):
        d_out, d_non_trainable = d.stateless_call(
            d_trainable, d_non_trainable, x, training=True
        )
        loss = weighted_bce(ops.squeeze(d_out, axis=-1), y, w)
        return loss, d_non_trainable

    def _gen_loss(g_trainable, g_non_trainable, d_trainable, d_non_trainable, z, x, y):
        w, g_non_trainable = _weights(g_trainable, g_non_trainable, z, y, training=True)
        d_out, _ = d.stateless_call(d_trainable, d_non_trainable, x, training=False)
        # g plays the opposite side of the same game: it maximizes the BCE that
        # d minimizes, so its loss is the negation.
        loss = -weighted_bce(ops.squeeze(d_out, axis=-1), y, w)
        return loss, g_non_trainable

    disc_grad_fn = jax.value_and_grad(_disc_loss, has_aux=True)
    gen_grad_fn = jax.value_and_grad(_gen_loss, has_aux=True)

    @jax.jit
    def disc_step(state: TrainState, z, x, y) -> tuple[TrainState, jax.Array]:
        """One discriminator update; g is frozen."""
        # Computed outside the differentiated function, so the weights are
        # constants here -- no stop_gradient needed to keep g out of the update.
        w, _ = _weights(state.g_trainable, state.g_non_trainable, z, y, training=False)
        (loss, d_non_trainable), grads = disc_grad_fn(
            state.d_trainable, state.d_non_trainable, x, y, w
        )
        d_trainable, opt_d_vars = opt_d.stateless_apply(
            state.opt_d, grads, state.d_trainable
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
    def gen_step(state: TrainState, z, x, y) -> tuple[TrainState, jax.Array]:
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
            state.opt_g, grads, state.g_trainable
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
    def eval_step(state: TrainState, z, x, y) -> jax.Array:
        """Weighted BCE with no updates, both models in inference mode."""
        w, _ = _weights(state.g_trainable, state.g_non_trainable, z, y, training=False)
        d_out, _ = d.stateless_call(
            state.d_trainable, state.d_non_trainable, x, training=False
        )
        return weighted_bce(ops.squeeze(d_out, axis=-1), y, w)

    return disc_step, gen_step, eval_step


def _as_batch(
    features: dict[str, npt.NDArray[np.double]], y: npt.NDArray[np.ubyte]
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Cast one dataset batch to the float64 arrays the steps expect."""
    return (
        features["z"].astype(np.double),
        features["x"].astype(np.double),
        y.reshape(-1).astype(np.double),
    )


def _eval_dataset(eval_step, state: TrainState, dataset: ArrayDataset) -> tuple[float, float]:
    """Mean weighted BCE over a split, as (d_loss, g_loss).

    g's loss is the exact negation of d's, so both entries report the same BCE;
    the pair is kept so the two curves stay directly comparable in the history.
    """
    total: float = 0.0
    n_batches: int = 0
    for features, y in dataset:
        loss = eval_step(state, *_as_batch(features, y))
        total += float(loss)
        n_batches += 1
    mean: float = total / n_batches
    return mean, mean


def _assign(variables: list[keras.Variable], values: Variables) -> None:
    """Write JAX arrays back into a model's `keras.Variable`s."""
    for var, val in zip(variables, values):
        var.assign(val)


def train(
    splits: DatasetSplits,
    dim: int = 1,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    patience: int = 5,
    min_delta: float = 1e-4,
    hidden_units: int = 64,
    n_layers: int = 2,
    seed: int | None = None,
) -> TrainResult:
    """Train the generator and discriminator.

    Arguments:
        seed: Weight-initialization seed. `None` draws one from system entropy.
            Either way the value used is returned, so a run stays reproducible
            after the fact without having to decide up front that it is worth
            reproducing.

    This seeds weight initialization *only*. The train/val/test split and the
    per-epoch batch order come from the dataset's own seed (`RAN_Dataset`),
    which draws from an independent generator. Varying `seed` across runs
    therefore estimates training/initialization variance at fixed data -- the
    usual HEP model-uncertainty ensemble -- while varying the dataset seed
    instead would fold in split variance.

    The networks are Dense-only with no dropout or batch norm and Adam is
    deterministic, so the two seeds together fully determine a run (up to
    non-deterministic GPU reductions).
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    keras.utils.set_random_seed(seed)
    # Rewind the batch-order sequence so repeated runs over one DatasetSplits
    # -- an ensemble loop over init seeds -- all see identical data.
    splits.train.reset()

    g: keras.Model = build_generator(dim=dim, hidden_units=hidden_units, n_layers=n_layers)
    d: keras.Model = build_discriminator(dim=dim, hidden_units=hidden_units, n_layers=n_layers)
    opt_g: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=lr_d)
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

    history: dict[str, list[float | np.floating]] = {"train_d": [], "train_g": [], "val_d": [], "val_g": [], }
    best_val_d: float = -np.inf
    best_state: TrainState | None = None
    wait: int = 0

    for epoch in range(n_epochs):
        d_losses: list[float] = []
        g_losses: list[float] = []
        for step, (features, y) in enumerate(splits.train):
            z, x, y_f = _as_batch(features, y)

            state, d_loss = disc_step(state, z, x, y_f)
            d_losses.append(float(d_loss))

            if step % n_disc_steps == 0:
                state, g_loss = gen_step(state, z, x, y_f)
                g_losses.append(-float(g_loss))

        mean_td: np.floating = np.mean(d_losses)
        mean_tg: np.floating = np.mean(g_losses)
        mean_val: tuple[float, float] = _eval_dataset(eval_step, state, splits.val)

        history["train_d"].append(mean_td)
        history["train_g"].append(mean_tg)
        history["val_d"].append(mean_val[0])
        history["val_g"].append(mean_val[1])

        # Early stopping: higher val D = better convergence toward log(2)
        if mean_val[0] > best_val_d + min_delta:
            best_val_d = mean_val[0]
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
                state = best_state
            break

    _assign(g.trainable_variables, state.g_trainable)
    _assign(g.non_trainable_variables, state.g_non_trainable)
    _assign(d.trainable_variables, state.d_trainable)
    _assign(d.non_trainable_variables, state.d_non_trainable)

    # Final test evaluation
    test: tuple[float, float] = _eval_dataset(eval_step, state, splits.test)
    logger.info("Test  D: %.4f  G: %.4f  (init seed %d)", test[0], test[1], seed)

    return TrainResult(g, d, history, seed)
