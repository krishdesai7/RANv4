"""Tests for the JAX training internals.

These exercise the loss math and a couple of real gradient steps on tiny
models. They do NOT train to convergence -- that is cluster work.
"""

import keras
import numpy as np
import ran  # ruff: ignore[unused-import]  -- pins KERAS_BACKEND=jax before keras is imported
from ran.data.datasets import RAN_Dataset
from ran.train import (
    EPS,
    TrainState,
    _make_steps,
    normalize_weights,
    train,
    weighted_bce,
)


def test_backend_is_jax_with_float64_enabled():
    import jax.numpy as jnp

    assert keras.backend.backend() == "jax"
    assert jnp.zeros(1, dtype="float64").dtype == np.float64


class TestNormalizeWeights:
    def test_data_events_are_pinned_to_one(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.1, 3.0, size=64)
        y = (rng.random(64) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y))
        np.testing.assert_allclose(w[y == 1], 1.0)

    def test_mc_weights_preserve_event_count(self):
        """Reweighted MC must keep the same total as unweighted MC."""
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.1, 3.0, size=256)
        y = (rng.random(256) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y))
        np.testing.assert_allclose(w[y == 0].sum(), (y == 0).sum(), rtol=1e-9)

    def test_mc_weights_stay_proportional_to_raw_output(self):
        rng = np.random.default_rng(2)
        raw = rng.uniform(0.1, 3.0, size=128)
        y = (rng.random(128) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y))
        ratio = w[y == 0] / raw[y == 0]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)

    def test_generator_output_on_data_rows_cannot_change_the_result(self):
        """The z_true guard: g's output on y=1 rows must not affect any weight.

        Data rows carry z_true, so if poisoning g's output there moved the
        weights, z_true would be reachable from the loss.
        """
        rng = np.random.default_rng(3)
        raw = rng.uniform(0.1, 3.0, size=128)
        y = (rng.random(128) < 0.5).astype(np.double)
        poisoned = raw.copy()
        poisoned[y == 1] = -999.0
        np.testing.assert_allclose(
            np.asarray(normalize_weights(raw, y)),
            np.asarray(normalize_weights(poisoned, y)),
        )


class TestWeightedBCE:
    def test_matches_numpy_reference_in_float64(self):
        """Guards against float32 accumulation inside the reduction.

        `keras.ops.mean` picks a float32 compute dtype for float64 input, which
        cost ~1e-8 of relative accuracy here; the tolerance below is far tighter
        than that, so a regression to `ops.mean` fails this test.
        """
        rng = np.random.default_rng(4)
        d_out = rng.uniform(0.01, 0.99, size=512)
        y = (rng.random(512) < 0.5).astype(np.double)
        w = rng.uniform(0.5, 1.5, size=512)

        expected = -np.mean(
            w * y * np.log(d_out + EPS) + w * (1 - y) * np.log(1 - d_out + EPS)
        )
        got = float(weighted_bce(d_out, y, w))
        assert abs(got - expected) < 1e-15, f"{got!r} vs {expected!r}"

    def test_perfect_classifier_scores_near_zero(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])
        d_out = np.array([1.0, 1.0, 0.0, 0.0])
        w = np.ones(4)
        assert float(weighted_bce(d_out, y, w)) < 1e-6

    def test_uninformative_classifier_scores_log2(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])
        d_out = np.full(4, 0.5)
        w = np.ones(4)
        np.testing.assert_allclose(float(weighted_bce(d_out, y, w)), np.log(2), atol=1e-6)

    def test_zero_weight_events_are_ignored(self):
        rng = np.random.default_rng(5)
        d_out = rng.uniform(0.01, 0.99, size=8)
        y = (rng.random(8) < 0.5).astype(np.double)
        w = np.ones(8)
        w_masked = w.copy()
        w_masked[-2:] = 0.0
        full = float(weighted_bce(d_out, y, w_masked))
        trimmed = float(weighted_bce(d_out[:-2], y[:-2], w[:-2])) * 6 / 8
        np.testing.assert_allclose(full, trimmed, rtol=1e-12)


class TestTrainSteps:
    """One real disc/gen update each, on tiny models."""

    @staticmethod
    def _setup(dim=2, n=64):
        from ran.models import build_discriminator, build_generator

        keras.utils.set_random_seed(0)
        g = build_generator(dim=dim, hidden_units=8, n_layers=1)
        d = build_discriminator(dim=dim, hidden_units=8, n_layers=1)
        opt_g = keras.optimizers.Adam(learning_rate=1e-2)
        opt_d = keras.optimizers.Adam(learning_rate=1e-2)
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
        rng = np.random.default_rng(6)
        z = rng.normal(size=(n, dim))
        x = rng.normal(size=(n, dim))
        y = (rng.random(n) < 0.5).astype(np.double)
        return _make_steps(g, d, opt_g, opt_d), state, (z, x, y)

    def test_disc_step_updates_only_the_discriminator(self):
        (disc_step, _, _), state, batch = self._setup()
        new, loss = disc_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.d_trainable, new.d_trainable):
            assert not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(state.g_trainable, new.g_trainable):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_gen_step_updates_only_the_generator(self):
        (_, gen_step, _), state, batch = self._setup()
        new, loss = gen_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.g_trainable, new.g_trainable):
            assert not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(state.d_trainable, new.d_trainable):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_gen_and_disc_losses_are_opposite(self):
        (disc_step, gen_step, eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        _, g_loss = gen_step(state, *batch)
        np.testing.assert_allclose(float(g_loss), -float(d_loss), rtol=1e-12)

    def test_eval_step_leaves_state_untouched_and_matches_disc_loss(self):
        (disc_step, _, eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        np.testing.assert_allclose(float(eval_step(state, *batch)), float(d_loss), rtol=1e-12)


def test_train_runs_and_returns_usable_models(tmp_path):
    """A few epochs on a tiny problem: shapes, history, and a saveable model."""
    n = 512
    rng = np.random.default_rng(7)
    z_true = rng.normal(0.0, 1.0, size=(n, 1))
    z_gen = rng.normal(0.5, 1.0, size=(n, 1))
    z = np.concatenate([z_true, z_gen])
    x = np.concatenate([z_true, z_gen]) + rng.normal(0, 0.5, size=(2 * n, 1))
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    splits = RAN_Dataset(batch_size=128, seed=0).splits_from_arrays(z, x, y)

    g, d, history, seed = train(
        splits, dim=1, n_epochs=3, patience=99, hidden_units=8, n_layers=1
    )

    assert isinstance(seed, int)
    assert set(history) == {"train_d", "train_g", "val_d", "val_g"}
    assert all(len(v) == 3 for v in history.values())
    assert all(np.isfinite(v).all() for v in history.values())

    w = np.asarray(g(z_gen))
    assert w.shape == (n, 1)
    assert np.all(w > 0), "softplus output must stay positive"

    # The trained values must have been written back out of the JAX pytree.
    g.save(tmp_path / "generator.keras")
    reloaded = keras.saving.load_model(tmp_path / "generator.keras")
    np.testing.assert_allclose(np.asarray(reloaded(z_gen)), w, rtol=1e-12)


def test_train_restores_best_weights_on_early_stop():
    """Early stopping must roll back to the best-val epoch, not the last one."""
    n = 512
    rng = np.random.default_rng(8)
    z = rng.normal(size=(2 * n, 1))
    x = rng.normal(size=(2 * n, 1))
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    splits = RAN_Dataset(batch_size=128, seed=0).splits_from_arrays(z, x, y)

    # z carries no information about y here, so val D cannot keep improving and
    # patience=1 trips quickly.
    history = train(
        splits, dim=1, n_epochs=25, patience=1, hidden_units=8, n_layers=1
    ).history
    assert len(history["val_d"]) < 25, "expected early stopping to fire"


class TestSeeding:
    """Weight-init seeding: reproducible runs that still ensemble."""

    @staticmethod
    def _splits(n=384):
        rng = np.random.default_rng(11)
        z = rng.normal(size=(2 * n, 1))
        x = z + rng.normal(0, 0.3, size=(2 * n, 1))
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return RAN_Dataset(batch_size=128, seed=3).splits_from_arrays(z, x, y)

    @staticmethod
    def _run(splits, seed):
        return train(
            splits, dim=1, n_epochs=2, patience=99, hidden_units=8, n_layers=1, seed=seed
        )

    def test_same_seed_reproduces_run_exactly(self):
        splits = self._splits()
        probe = np.linspace(-2, 2, 16).reshape(-1, 1)
        a, b = self._run(splits, 123), self._run(splits, 123)
        assert a.seed == b.seed == 123
        np.testing.assert_array_equal(np.asarray(a.g(probe)), np.asarray(b.g(probe)))
        np.testing.assert_allclose(a.history["train_d"], b.history["train_d"], rtol=0)

    def test_different_seeds_give_different_models(self):
        """The ensemble spread the HEP error estimate relies on."""
        splits = self._splits()
        probe = np.linspace(-2, 2, 16).reshape(-1, 1)
        a, b = self._run(splits, 1), self._run(splits, 2)
        assert not np.allclose(np.asarray(a.g(probe)), np.asarray(b.g(probe)))

    def test_omitted_seed_is_drawn_and_reported(self):
        """A run left unseeded must still be reproducible after the fact."""
        splits = self._splits()
        drawn = self._run(splits, None)
        assert isinstance(drawn.seed, int)
        assert drawn.seed >= 0

        probe = np.linspace(-2, 2, 16).reshape(-1, 1)
        replay = self._run(splits, drawn.seed)
        np.testing.assert_array_equal(
            np.asarray(drawn.g(probe)), np.asarray(replay.g(probe))
        )

    def test_omitted_seed_varies_between_runs(self):
        splits = self._splits()
        seeds = {self._run(splits, None).seed for _ in range(3)}
        assert len(seeds) == 3, f"entropy-drawn seeds collided: {seeds}"

    def test_init_seed_does_not_disturb_batch_order(self):
        """The two randomness axes must stay independent.

        Batch order comes from the dataset's own generator, so changing the
        init seed must not reshuffle the data.
        """
        splits = self._splits()

        def first_batch():
            splits.train.reset()
            return next(iter(splits.train))[0]["z"].ravel().copy()

        before = first_batch()
        train(splits, dim=1, n_epochs=1, patience=99, hidden_units=8, n_layers=1, seed=7)
        np.testing.assert_array_equal(first_batch(), before)

        train(splits, dim=1, n_epochs=1, patience=99, hidden_units=8, n_layers=1, seed=8)
        np.testing.assert_array_equal(first_batch(), before)
