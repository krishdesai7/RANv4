from __future__ import annotations

import tensorflow as tf
import keras

from datasets import RAN_Dataset, DatasetSplits
from models import build_generator, build_discriminator

EPS = 1e-7


def _compute_weights(g: keras.Model, z: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Return per-event weights: 1 for data, normalized g(z) for MC."""
    raw_w = tf.squeeze(g(z), axis=-1)
    n_mc = tf.reduce_sum(1.0 - y)
    w_mc_norm = raw_w * n_mc / (tf.reduce_sum(raw_w * (1.0 - y)) + EPS)
    return y + (1.0 - y) * w_mc_norm


def _weighted_bce(d_out: tf.Tensor, y: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    """Weighted binary cross-entropy."""
    return -tf.reduce_mean(
        w * y * tf.math.log(d_out + EPS)
        + w * (1.0 - y) * tf.math.log(1.0 - d_out + EPS)
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
        d_out = tf.squeeze(d(x, training=True), axis=-1)
        loss = _weighted_bce(d_out, y, w)
    grads = tape.gradient(loss, d.trainable_variables)
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
        w = _compute_weights(g, z, y)
        d_out = tf.squeeze(d(x, training=False), axis=-1)
        loss = -_weighted_bce(d_out, y, w)
    grads = tape.gradient(loss, g.trainable_variables)
    opt_g.apply_gradients(zip(grads, g.trainable_variables))
    return loss


def _eval_step(
    g: keras.Model,
    d: keras.Model,
    z: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tuple[float, float]:
    w = _compute_weights(g, z, y)
    d_out = tf.squeeze(d(x, training=False), axis=-1)
    d_loss = float(_weighted_bce(d_out, y, w))
    g_loss = -d_loss
    return d_loss, g_loss


def train(
    n_epochs: int = 10,
    n_disc_steps: int = 5,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    batch_size: int = 1024,
    n_samples: int = 500_000,
    smearing: float = 1.0,
) -> tuple[keras.Model, keras.Model, DatasetSplits, dict[str, list[float]]]:
    splits = RAN_Dataset(batch_size=batch_size).generate_gaussian_dataset(
        n_samples=n_samples, smearing=smearing,
    )
    g = build_generator()
    d = build_discriminator()
    opt_g = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d = keras.optimizers.Adam(learning_rate=lr_d)

    history: dict[str, list[float]] = {
        "train_d": [], "train_g": [], "test_d": [], "test_g": [],
    }

    for epoch in range(n_epochs):
        d_losses: list[float] = []
        g_losses: list[float] = []
        for step, (features, y) in enumerate(splits.train):
            z = tf.cast(features["z"], tf.float32)
            x = tf.cast(features["x"], tf.float32)
            y_f = tf.cast(tf.squeeze(y, axis=-1), tf.float32)

            d_loss = _disc_step(g, d, opt_d, z, x, y_f)
            d_losses.append(float(d_loss))

            if step % n_disc_steps == 0:
                g_loss = _gen_step(g, d, opt_g, z, x, y_f)
                g_losses.append(float(-g_loss))

        test_d: list[float] = []
        test_g: list[float] = []
        for features, y in splits.test:
            z = tf.cast(features["z"], tf.float32)
            x = tf.cast(features["x"], tf.float32)
            y_f = tf.cast(tf.squeeze(y, axis=-1), tf.float32)
            d_l, g_l = _eval_step(g, d, z, x, y_f)
            test_d.append(d_l)
            test_g.append(-g_l)

        mean_td = sum(d_losses) / len(d_losses)
        mean_tg = sum(g_losses) / len(g_losses)
        mean_vd = sum(test_d) / len(test_d)
        mean_vg = sum(test_g) / len(test_g)
        history["train_d"].append(mean_td)
        history["train_g"].append(mean_tg)
        history["test_d"].append(mean_vd)
        history["test_g"].append(mean_vg)

        print(
            f"Epoch {epoch + 1:3d}/{n_epochs}"
            f"  D: {mean_td:.4f}  G: {mean_tg:.4f}"
            f"  | Test D: {mean_vd:.4f}  G: {mean_vg:.4f}"
        )

    return g, d, splits, history
