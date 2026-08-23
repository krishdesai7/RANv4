"""Tests for the JAX training internals.

These exercise the loss math and a couple of real gradient steps on tiny
models. They do NOT train to convergence -- that is cluster work.
"""

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
from numpy import dtype, float64, ndarray
from ran.data.datasets import DatasetSplits, RANDataset
from ran.data.device import DeviceSplits, train_indices
from ran.models import build_generator
from ran.rantypes import TRUTH_SENTINEL, ZXY, Events
from ran.train import (
    EPS,
    TrainResult,
    TrainState,
    _make_steps,
    bce_sums,
    normalize_weights,
    train,
    weighted_bce,
)


def ones(n: int) -> ndarray[tuple[int], dtype[np.single]]:
    """An all-ones mask: what the training path always passes."""
    return np.ones(n, dtype=np.single)


def test_backend_is_jax_pinned_to_single_precision() -> None:
    """The pin is a package-import side effect, so it needs a guard.

    `JAX_ENABLE_X64=0` means a float64 request is silently truncated rather
    than refused, which is exactly the kind of thing that goes unnoticed until
    a run's numbers drift. Asserting the truncation makes the policy explicit.
    """
    import jax.numpy as jnp

    assert keras.backend.backend() == "jax"
    assert jnp.zeros(1).dtype == np.float32
    with pytest.warns(UserWarning, match="float64"):
        assert jnp.zeros(1, dtype="float64").dtype == np.float32


class TestNormalizeWeights:
    def test_data_events_are_pinned_to_one(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.1, 3.0, size=64)
        y = (rng.random(64) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y, ones(raw.size)))
        np.testing.assert_allclose(w[y == 1], 1.0)

    def test_mc_weights_preserve_event_count(self) -> None:
        """Reweighted MC must keep the same total as unweighted MC."""
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.1, 3.0, size=256)
        y = (rng.random(256) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y, ones(raw.size)))
        np.testing.assert_allclose(w[y == 0].sum(), (y == 0).sum(), rtol=1e-6)

    def test_mc_weights_stay_proportional_to_raw_output(self) -> None:
        rng = np.random.default_rng(2)
        raw = rng.uniform(0.1, 3.0, size=128)
        y = (rng.random(128) < 0.5).astype(np.double)
        w = np.asarray(normalize_weights(raw, y, ones(raw.size)))
        ratio = w[y == 0] / raw[y == 0]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)

    def test_generator_output_on_data_rows_cannot_change_the_result(self) -> None:
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
            np.asarray(normalize_weights(raw, y, ones(raw.size))),
            np.asarray(normalize_weights(poisoned, y, ones(poisoned.size))),
        )


def test_a_missing_particle_level_cannot_poison_the_batch() -> None:
    """Why `TRUTH_SENTINEL` is a number: the z_true guard is arithmetic.

    A sample built by `Populations.create` without truth carries the sentinel
    in the nature rows of `z`, so g forward-passes it. Multiplying g's output
    there by (1 - y) = 0 annihilates a number, which is what makes those rows
    irrelevant -- but under IEEE 754 it does not annihilate a NaN, which would
    instead reach every weight in the batch through the normalizing sum.
    """
    rng = np.random.default_rng(6)
    mc_z = rng.normal(size=(6, 1)).astype(np.single)
    y = np.array([1.0] * 4 + [0.0] * 6)
    g = build_generator(dim=1)

    def weights(nature_fill: float) -> ndarray[tuple[int, ...], dtype[float64]]:
        z = np.concatenate([np.full((4, 1), nature_fill), mc_z], axis=0)
        raw = np.squeeze(np.asarray(g(z)), axis=-1)
        return np.asarray(normalize_weights(raw, y, ones(raw.size)))

    np.testing.assert_allclose(weights(float(TRUTH_SENTINEL)), weights(0.0))
    assert np.all(np.isnan(weights(np.nan)))


class TestWeightedBCE:
    def test_matches_a_float64_reference_to_single_precision(self) -> None:
        """The reduction must not lose more than single-precision rounding.

        Scored against a float64 numpy reference: 512 terms accumulated well
        stay within a few float32 ULPs of it, while a sloppily ordered or
        narrower-than-float32 accumulation drifts further. This is the test that
        pinned `jnp.sum(...) / n` over a mean back when `keras.ops.mean` picked
        a float32 compute dtype for float64 input; the hazard is gone with the
        pipeline at float32, but the reduction is still the thing being pinned.
        """
        rng = np.random.default_rng(4)
        d_out = rng.uniform(0.01, 0.99, size=512).astype(np.single)
        y = (rng.random(512) < 0.5).astype(np.single)
        w = rng.uniform(0.5, 1.5, size=512).astype(np.single)

        wide = (d_out.astype(np.double), y.astype(np.double), w.astype(np.double))
        expected = -np.mean(
            wide[2] * wide[1] * np.log(wide[0] + EPS)
            + wide[2] * (1 - wide[1]) * np.log(1 - wide[0] + EPS)
        )
        got = float(weighted_bce(d_out, y, w, ones(d_out.size)))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_perfect_classifier_scores_near_zero(self) -> None:
        y = np.array([1.0, 1.0, 0.0, 0.0])
        d_out = np.array([1.0, 1.0, 0.0, 0.0])
        w = np.ones(4)
        assert float(weighted_bce(d_out, y, w, ones(d_out.size))) < 1e-6

    def test_uninformative_classifier_scores_log2(self) -> None:
        y = np.array([1.0, 1.0, 0.0, 0.0])
        d_out = np.full(4, 0.5)
        w = np.ones(4)
        np.testing.assert_allclose(
            float(weighted_bce(d_out, y, w, ones(d_out.size))), np.log(2), atol=1e-6
        )

    def test_masked_rows_are_ignored_outright(self) -> None:
        """Zeroing `mask` must equal deleting the rows, denominator included.

        This is what makes a padded eval batch report the number an unpadded
        one would: a zero weight still counts toward the mean, a zero mask does
        not.
        """
        rng = np.random.default_rng(9)
        d_out = rng.uniform(0.01, 0.99, size=8)
        y = (rng.random(8) < 0.5).astype(np.double)
        w = rng.uniform(0.5, 1.5, size=8)
        mask = ones(8)
        mask[-2:] = 0.0
        np.testing.assert_allclose(
            float(weighted_bce(d_out, y, w, mask)),
            float(weighted_bce(d_out[:-2], y[:-2], w[:-2], ones(6))),
            rtol=1e-6,
        )

    def test_sums_divide_to_the_mean(self) -> None:
        """`bce_sums` is the reduction split in two, so a scan can accumulate."""
        rng = np.random.default_rng(10)
        d_out = rng.uniform(0.01, 0.99, size=16)
        y = (rng.random(16) < 0.5).astype(np.double)
        w = rng.uniform(0.5, 1.5, size=16)
        total, count = bce_sums(d_out, y, w, ones(16))
        assert int(count) == 16
        np.testing.assert_allclose(
            float(total) / float(count),
            float(weighted_bce(d_out, y, w, ones(16))),
            rtol=1e-6,
        )

    def test_summing_batches_matches_scoring_them_together(self) -> None:
        """Accumulating (total, count) across batches gives the true mean --
        not the mean of per-batch means the old host loop computed."""
        rng = np.random.default_rng(12)
        d_out = rng.uniform(0.01, 0.99, size=10)
        y = (rng.random(10) < 0.5).astype(np.double)
        w = rng.uniform(0.5, 1.5, size=10)
        halves = [
            bce_sums(d_out[s], y[s], w[s], ones(len(y[s])))
            for s in (slice(0, 6), slice(6, 10))
        ]
        combined = sum(float(t) for t, _ in halves) / sum(float(c) for _, c in halves)
        np.testing.assert_allclose(
            combined, float(weighted_bce(d_out, y, w, ones(10))), rtol=1e-6
        )

    def test_zero_weight_events_are_ignored(self) -> None:
        rng = np.random.default_rng(5)
        d_out = rng.uniform(0.01, 0.99, size=8)
        y = (rng.random(8) < 0.5).astype(np.double)
        w = np.ones(8)
        w_masked = w.copy()
        w_masked[-2:] = 0.0
        full = float(weighted_bce(d_out, y, w_masked, ones(8)))
        trimmed = float(weighted_bce(d_out[:-2], y[:-2], w[:-2], ones(6))) * 6 / 8
        np.testing.assert_allclose(full, trimmed, rtol=1e-6)


class TestTrainSteps:
    """One real disc/gen update each, on tiny models."""

    @staticmethod
    def _setup(dim: int = 2, n: int = 64):
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
        # Device arrays, because that is what the steps see: they are traced
        # inside the epoch program, never called on host NumPy. Labels arrive
        # already promoted to the compute dtype -- `TrainSplit` does that once,
        # on the way to device, rather than per `1 - y`.
        batch = tuple(
            jnp.asarray(a)
            for a in (
                rng.normal(size=(n, dim)).astype(np.single),
                rng.normal(size=(n, dim)).astype(np.single),
                (rng.random(n) < 0.5).astype(np.double),
                # On the training path the mask is always all ones.
                ones(n),
            )
        )
        return _make_steps(g, d, opt_g, opt_d), state, batch

    def test_disc_step_updates_only_the_discriminator(self) -> None:
        (disc_step, _, _), state, batch = self._setup()
        new, loss = disc_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.d_trainable, new.d_trainable, strict=False):
            assert not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(state.g_trainable, new.g_trainable, strict=False):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_gen_step_updates_only_the_generator(self) -> None:
        (_, gen_step, _), state, batch = self._setup()
        new, loss = gen_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.g_trainable, new.g_trainable, strict=False):
            assert not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(state.d_trainable, new.d_trainable, strict=False):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_gen_and_disc_losses_are_opposite(self) -> None:
        (disc_step, gen_step, _eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        _, g_loss = gen_step(state, *batch)
        np.testing.assert_allclose(float(g_loss), -float(d_loss), rtol=1e-6)

    def test_eval_step_leaves_state_untouched_and_matches_disc_loss(self) -> None:
        (disc_step, _, eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        # eval_step hands back the two halves unreduced, so a scan can add them
        # up across batches and divide once.
        total, count = eval_step(state, *batch)
        np.testing.assert_allclose(
            float(total) / float(count), float(d_loss), rtol=1e-6
        )
        assert int(count) == len(batch[2])


def test_train_runs_and_returns_usable_models(tmp_path) -> None:
    """A few epochs on a tiny problem: shapes, history, and a saveable model."""
    n = 512
    rng = np.random.default_rng(7)
    z_true = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.single)
    z_gen = rng.normal(0.5, 1.0, size=(n, 1)).astype(np.single)
    z = np.concatenate([z_true, z_gen])
    x = np.concatenate([z_true, z_gen]) + rng.normal(0, 0.5, size=(2 * n, 1)).astype(
        np.single
    )
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    splits = RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y))

    g, _d, history, seed = train(
        splits, dim=1, n_epochs=3, patience=99, hidden_units=8, n_layers=1, seed=None
    )

    assert isinstance(seed, int)
    assert set(history) == {"train_d", "train_g", "val_d", "val_g"}
    # History values are only SupportsFloat, as the rest of the pipeline reads
    # them: `plot_losses` and `_save_run` both name a dtype to convert them.
    curves = {k: np.array(v, dtype=np.single) for k, v in history.items()}
    assert all(len(v) == 3 for v in curves.values())
    assert all(np.isfinite(v).all() for v in curves.values())

    w = np.asarray(g(z_gen))
    assert w.shape == (n, 1)
    assert np.all(w > 0), "softplus output must stay positive"

    # The trained values must have been written back out of the JAX pytree.
    g.save(tmp_path / "generator.keras")
    reloaded = keras.saving.load_model(tmp_path / "generator.keras")
    np.testing.assert_allclose(np.asarray(reloaded(z_gen)), w, rtol=1e-6)


def test_train_restores_best_weights_on_early_stop() -> None:
    """Early stopping must roll back to the best-val epoch, not the last one."""
    n = 512
    rng = np.random.default_rng(8)
    z = rng.normal(size=(2 * n, 1)).astype(np.single)
    x = rng.normal(size=(2 * n, 1)).astype(np.single)
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    splits = RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y))

    # z carries no information about y here, so val D cannot keep improving and
    # patience=1 trips quickly.
    history = train(
        splits, dim=1, n_epochs=25, patience=1, hidden_units=8, n_layers=1, seed=None
    ).history
    assert len(history["val_d"]) < 25, "expected early stopping to fire"


class TestSeeding:
    """Weight-init seeding: reproducible runs that still ensemble."""

    @staticmethod
    def _splits(n: int = 384) -> DatasetSplits:
        rng = np.random.default_rng(11)
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = z + rng.normal(0, 0.3, size=(2 * n, 1)).astype(np.single)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return RANDataset(batch_size=128, seed=3).splits_from_data(ZXY(Events(z, x), y))

    @staticmethod
    def _run(splits: DatasetSplits, seed: int | None) -> TrainResult:
        return train(
            splits,
            dim=1,
            n_epochs=2,
            patience=99,
            hidden_units=8,
            n_layers=1,
            seed=seed,
        )

    def test_same_seed_reproduces_run_exactly(self) -> None:
        splits = self._splits()
        probe = np.linspace(-2, 2, 16).reshape(-1, 1)
        a, b = self._run(splits, 123), self._run(splits, 123)
        assert a.seed == b.seed == 123
        np.testing.assert_array_equal(np.asarray(a.g(probe)), np.asarray(b.g(probe)))
        np.testing.assert_allclose(
            np.array(a.history["train_d"], dtype=np.single),
            np.array(b.history["train_d"], dtype=np.single),
            rtol=0,
        )

    def test_different_seeds_give_different_models(self) -> None:
        """The ensemble spread the HEP error estimate relies on."""
        splits = self._splits()
        probe = np.linspace(-2, 2, 16).reshape(-1, 1)
        a, b = self._run(splits, 1), self._run(splits, 2)
        assert not np.allclose(np.asarray(a.g(probe)), np.asarray(b.g(probe)))

    def test_omitted_seed_is_drawn_and_reported(self) -> None:
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

    def test_omitted_seed_varies_between_runs(self) -> None:
        splits = self._splits()
        seeds = {self._run(splits, None).seed for _ in range(3)}
        assert len(seeds) == 3, f"entropy-drawn seeds collided: {seeds}"

    def test_init_seed_does_not_disturb_batch_order(self) -> None:
        """The two randomness axes must stay independent.

        Batch order is drawn from the split's own `data_seed`, so changing the
        init seed must not reshuffle the data. This is what makes the HEP
        ensemble -- a loop over `--seed` at fixed `--data_seed` -- measure
        initialization variance and nothing else.
        """
        splits = self._splits()
        curves = [
            train(
                splits,
                dim=1,
                n_epochs=2,
                patience=99,
                hidden_units=8,
                n_layers=1,
                seed=s,
            )
            for s in (7, 8)
        ]
        # Different inits, so the losses differ -- but both runs consumed the
        # same batches, which is what the shared key guarantees.
        assert curves[0].seed != curves[1].seed
        key = jax.random.key(splits.train.seed)
        np.testing.assert_array_equal(
            np.asarray(train_indices(key, splits.train.size, 128, 5)),
            np.asarray(train_indices(key, splits.train.size, 128, 5)),
        )


class TestFusion:
    """The fused whole-run program against the Python-driven reference."""

    @staticmethod
    def _splits(n: int = 512) -> DatasetSplits:
        rng = np.random.default_rng(21)
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = z + rng.normal(0, 0.3, size=(2 * n, 1)).astype(np.single)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return RANDataset(batch_size=32, seed=5).splits_from_data(ZXY(Events(z, x), y))

    def test_fused_and_eager_runs_agree(self) -> None:
        """`fused=False` is the debugging path, so it must not be a second model.

        It runs the identical epoch function from a Python `while` instead of a
        `lax.while_loop`, so the two differ only in how XLA is allowed to
        associate the reductions -- not in the arithmetic.
        """
        splits = self._splits()
        kwargs = {
            "dim": 1,
            "n_epochs": 3,
            "patience": 99,
            "hidden_units": 8,
            "n_layers": 1,
            "seed": 42,
        }
        fused = train(splits, fused=True, **kwargs)
        eager = train(splits, fused=False, **kwargs)
        for key in ("train_d", "train_g", "val_d", "val_g"):
            np.testing.assert_allclose(
                fused.history[key], eager.history[key], rtol=1e-10
            )

    def test_padded_eval_matches_an_exactly_divisible_one(self) -> None:
        """Padding rows must be inert.

        The eval splits are padded to a whole number of batches and the filler
        carries `mask == 0`; it enters no sum, so the reported loss is the one an
        unpadded split would have given.
        """
        splits = self._splits()
        n_val = splits.val.size
        assert n_val % 7 != 0, "want a size that does not divide evenly"
        exact = DeviceSplits.from_splits(splits, eval_batch_size=n_val)
        padded = DeviceSplits.from_splits(splits, eval_batch_size=7)
        assert padded.val.n_batches > exact.val.n_batches
        assert int(padded.val.mask.sum()) == n_val
        assert int(exact.val.mask.sum()) == n_val
