import tensorflow as tf
import keras

from datasets import RAN_Dataset, DatasetSplits
from models import build_generator, build_discriminator

EPS = keras.config.epsilon()


def _compute_weights(g: keras.Model, z: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Return per-event weights: 1 for data, normalized g(z) for MC.
    """
    raw_w: tf.Tensor = tf.squeeze(g(z), axis=-1)
    n_mc: tf.Tensor = tf.reduce_sum(1.0 - y)
    w_mc_norm: tf.Tensor = raw_w * n_mc / (tf.reduce_sum(raw_w * (1.0 - y)) + EPS)
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
) -> tuple[float, float]:
    w = _compute_weights(g, z, y)
    d_out = tf.squeeze(d(x, training=False), axis=-1)
    d_loss = float(_weighted_bce(d_out, y, w))
    g_loss = -d_loss
    return d_loss, g_loss


def _eval_dataset(
    g: keras.Model,
    d: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[float, float]:
    d_vals: list[float] = []
    g_vals: list[float] = []
    for features, y in dataset:
        z = tf.cast(features["z"], tf.float32)
        x = tf.cast(features["x"], tf.float32)
        y_f = tf.cast(tf.squeeze(y, axis=-1), tf.float32)
        d_l, g_l = _eval_step(g, d, z, x, y_f)
        d_vals.append(d_l)
        g_vals.append(-g_l)
    return sum(d_vals) / len(d_vals), sum(g_vals) / len(g_vals)


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
) -> tuple[keras.Model, keras.Model, DatasetSplits, dict[str, list[float]]]:
    splits = RAN_Dataset(batch_size=batch_size).generate_gaussian_dataset(
        n_samples=n_samples, smearing=smearing,
    )
    g = build_generator()
    d = build_discriminator()
    opt_g = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d = keras.optimizers.Adam(learning_rate=lr_d)

    history: dict[str, list[float]] = {
        "train_d": [], "train_g": [], "val_d": [], "val_g": [],
    }

    best_val_d = -float("inf")
    best_g_weights: list | None = None
    best_d_weights: list | None = None
    wait = 0

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

        mean_td = sum(d_losses) / len(d_losses)
        mean_tg = sum(g_losses) / len(g_losses)
        mean_vd, mean_vg = _eval_dataset(g, d, splits.val)

        history["train_d"].append(mean_td)
        history["train_g"].append(mean_tg)
        history["val_d"].append(mean_vd)
        history["val_g"].append(mean_vg)

        # Early stopping: higher val D = better convergence toward log(2)
        if mean_vd > best_val_d + min_delta:
            best_val_d = mean_vd
            best_g_weights = [w.numpy().copy() for w in g.trainable_variables]
            best_d_weights = [w.numpy().copy() for w in d.trainable_variables]
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch + 1:3d}/{n_epochs}"
            f"  D: {mean_td:.4f}  G: {mean_tg:.4f}"
            f"  | Val D: {mean_vd:.4f}  G: {mean_vg:.4f}"
            f"  (patience {wait}/{patience})"
        )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            for var, val in zip(g.trainable_variables, best_g_weights):
                var.assign(val)
            for var, val in zip(d.trainable_variables, best_d_weights):
                var.assign(val)
            break

    # Final test evaluation
    test_d, test_g = _eval_dataset(g, d, splits.test)
    print(f"Test  D: {test_d:.4f}  G: {test_g:.4f}")

    return g, d, splits, history
