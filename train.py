import tensorflow as tf
import keras
import numpy as np

from datasets import RAN_Dataset, DatasetSplits
from models import build_generator, build_discriminator

EPS = keras.config.epsilon()


def _compute_weights(g: keras.Model, z: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Return per-event weights: 1 for data, normalized g(z) for MC.
    """
    raw_w: tf.Tensor = tf.squeeze(g(z), axis=-1)
    y = tf.cast(y, raw_w.dtype)
    one = tf.ones_like(y)
    eps = tf.cast(EPS, raw_w.dtype)
    n_mc: tf.Tensor = tf.reduce_sum(one - y)
    w_mc_norm: tf.Tensor = raw_w * n_mc / (tf.reduce_sum(raw_w * (one - y)) + eps)
    return y + (one - y) * w_mc_norm


def _weighted_bce(d_out: tf.Tensor, y: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    """Weighted binary cross-entropy."""
    y = tf.cast(y, d_out.dtype)
    w = tf.cast(w, d_out.dtype)
    one = tf.ones_like(d_out)
    eps = tf.cast(EPS, d_out.dtype)
    return -tf.reduce_mean(
        w * y * tf.math.log(d_out + eps)
        + w * (one - y) * tf.math.log(one - d_out + eps)
    )


@tf.function
def _disc_step(
    g: keras.Model,
    d: keras.Model,
    opt_d: keras.optimizers.Optimizer,
    z: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tf.Tensor:
    w = tf.stop_gradient(_compute_weights(g, z, y))
    with tf.GradientTape() as tape:
        d_out: tf.Tensor = tf.squeeze(d(x, training=True), axis=-1)
        loss: tf.Tensor = _weighted_bce(d_out, y, w)
    grads: list[tf.Gradients] = tape.gradient(loss, d.trainable_variables)
    opt_d.apply_gradients(zip(grads, d.trainable_variables))
    return loss


@tf.function
def _gen_step(
    g: keras.Model,
    d: keras.Model,
    opt_g: keras.optimizers.Optimizer,
    z: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        w: tf.Tensor = _compute_weights(g, z, y)
        d_out: tf.Tensor = tf.squeeze(d(x, training=False), axis=-1)
        loss: tf.Tensor = -_weighted_bce(d_out, y, w)
    grads: list[tf.Gradients] = tape.gradient(loss, g.trainable_variables)
    opt_g.apply_gradients(zip(grads, g.trainable_variables))
    return loss


def _eval_step(
    g: keras.Model,
    d: keras.Model,
    z: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tuple[np.double, np.double]:
    w: tf.Tensor = _compute_weights(g, z, y)
    d_out: tf.Tensor = tf.squeeze(d(x, training=False), axis=-1)
    d_loss: np.double = _weighted_bce(d_out, y, w).numpy()
    g_loss: np.double = -d_loss
    return d_loss, g_loss


def _eval_dataset(
    g: keras.Model,
    d: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[float, float]:
    d_sum: float = 0.0
    g_sum: float = 0.0
    n_batches: int = 0
    for features, y in dataset:
        z: tf.Tensor = tf.cast(features["z"], tf.double)
        x: tf.Tensor = tf.cast(features["x"], tf.double)
        y_f: tf.Tensor = tf.cast(tf.squeeze(y, axis=-1), tf.double)
        g_l: np.double
        d_l: np.double
        d_l, g_l = _eval_step(g, d, z, x, y_f)
        d_sum += d_l
        g_sum -= g_l
        n_batches += 1
    return d_sum / n_batches, g_sum / n_batches


def train(
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    batch_size: int = 1024,
    n_samples: int = 500_000,
    smearing: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-4,
) -> tuple[keras.Model, keras.Model, DatasetSplits, dict[str, list[float | np.floating]]]:
    """Train the generator and discriminator.
    Arguments:
        n_epochs (int)
        n_disc_steps (int)
        lr_g (float)
        lr_d (float)
        batch_size (int)
        n_samples (int)
        smearing (float)
        patience (int)
        min_delta (float)
    Returns:
        tuple[
            g (keras.Model): Generator model.
            d (keras.Model): Discriminator model.
            splits (DatasetSplits)
            history (dict[str, list[float | np.floating]]): Training history.
        ]
    """
    splits: DatasetSplits = RAN_Dataset(
        batch_size=batch_size
        ).generate_gaussian_dataset(
        n_samples=n_samples,
        smearing=smearing,
    )
    g: keras.Model = build_generator()
    d: keras.Model = build_discriminator()
    opt_g: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=lr_d)
    history: dict[str, list[float | np.floating]] = { "train_d": [], "train_g": [], "val_d": [], "val_g": [], }
    best_val_d: float = -np.inf
    best_g_weights: list | None = None
    best_d_weights: list | None = None
    wait: int = 0

    for epoch in range(n_epochs):
        d_losses: list[np.double] = []
        g_losses: list[np.double] = []
        for step, (features, y) in enumerate(splits.train):
            z: tf.Tensor = tf.cast(features["z"], tf.double)
            x: tf.Tensor = tf.cast(features["x"], tf.double)
            y_f: tf.Tensor = tf.cast(tf.squeeze(y, axis=-1), tf.double)

            d_loss: tf.Tensor = _disc_step(g, d, opt_d, z, x, y_f)
            d_losses.append(d_loss.numpy())

            if step % n_disc_steps == 0:
                g_loss: tf.Tensor = _gen_step(g, d, opt_g, z, x, y_f)
                g_losses.append(-g_loss.numpy())

        mean_td: np.floating = np.mean(d_losses)
        mean_tg: np.floating = np.mean(g_losses)
        mean_val: tuple[float, float] = _eval_dataset(g, d, splits.val)

        history["train_d"].append(mean_td)
        history["train_g"].append(mean_tg)
        history["val_d"].append(mean_val[0])
        history["val_g"].append(mean_val[1])

        # Early stopping: higher val D = better convergence toward log(2)
        if mean_val[0] > best_val_d + min_delta:
            best_val_d = mean_val[0]
            best_g_weights = [w.numpy().copy() for w in g.trainable_variables]
            best_d_weights = [w.numpy().copy() for w in d.trainable_variables]
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch + 1:3d}/{n_epochs}"
            f"  D: {mean_td:.4f}  G: {mean_tg:.4f}"
            f"  | Val D: {mean_val[0]:.4f}  G: {mean_val[1]:.4f}"
            f"  (patience {wait}/{patience})"
        )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            for var, val in zip(g.trainable_variables, best_g_weights or []):
                var.assign(val)
            for var, val in zip(d.trainable_variables, best_d_weights or []):
                var.assign(val)
            break

    # Final test evaluation
    test: tuple[float, float] = _eval_dataset(g, d, splits.test)
    print(f"Test  D: {test[0]:.4f}  G: {test[1]:.4f}")

    return g, d, splits, history
