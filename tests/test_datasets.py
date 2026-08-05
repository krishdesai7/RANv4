from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from numpy import dtype, float64, ndarray
from ran.data.datasets import ArrayDataset, DatasetSplits, RANDataset

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(params: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(params))
    return p


class TestGenerateGaussianDataset:
    """Test multivariate Gaussian dataset generation."""

    def test_1d_uncorrelated(self, tmp_path) -> None:
        """1D scalar sigma should produce valid splits."""
        cfg = {
            "mu_gen": [0.5],
            "mu_true": [0.0],
            "sigma_gen": 0.9,
            "sigma_true": 1.0,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        ds = RANDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=1000)
        assert splits.train is not None
        assert splits.val is not None
        assert splits.test is not None

    def test_2d_correlated_shapes(self, tmp_path) -> None:
        """2D with full covariance should produce correct shapes."""
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [[1.0, -0.54], [-0.54, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        ds = RANDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=2000)
        for features, _y in splits.test:
            assert features["z"].shape[1] == 2
            assert features["x"].shape[1] == 2
            break

    def test_params_dict_interface(self) -> None:
        """Passing params dict directly should work (for --load_run)."""
        params = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        ds = RANDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(params=params, n_samples=1000)
        assert splits.train is not None

    def test_both_config_and_params_raises(self, tmp_path) -> None:
        """Providing both config_path and params should error."""
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        ds = RANDataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(config_path=path, params=cfg, n_samples=100)

    def test_neither_config_nor_params_raises(self) -> None:
        """Providing neither config_path nor params should error."""
        ds = RANDataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(n_samples=100)

    def test_caching(self, tmp_path) -> None:
        """Second call with same config should hit cache."""
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"
        ds = RANDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files = list(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files) == 1
        ds2 = RANDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds2.generate_gaussian_dataset(config_path=path, n_samples=500)

    def test_smearing_preserves_event_coupling(self, tmp_path) -> None:
        """Detector-level values should be correlated with particle-level."""
        cfg = {
            "mu_gen": [0.0, 0.0],
            "mu_true": [0.0, 0.0],
            "sigma_gen": [1.0, 1.0],
            "sigma_true": [1.0, 1.0],
            "sigma_detector": [0.1, 0.1],
        }
        path = _write_config(cfg, tmp_path)
        ds = RANDataset(batch_size=10000, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=10000)
        for features, _y in splits.test:
            z = features["z"]
            x = features["x"]
            for d in range(2):
                corr = np.corrcoef(z[:, d], x[:, d])[0, 1]
                assert corr > 0.95, f"dim {d}: corr={corr}, expected >0.95"
            break

    def test_yaml_and_params_share_cache(self, tmp_path) -> None:
        """YAML path and equivalent params dict must produce the same cache key."""
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"

        ds1 = RANDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds1.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files_after_yaml = set(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files_after_yaml) == 1

        reload_params = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [[1.0, 0.0], [0.0, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [[0.25, 0.0], [0.0, 0.64]],
        }
        ds2 = RANDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds2.generate_gaussian_dataset(params=reload_params, n_samples=500)

        cache_files_after_params = set(cache_dir.glob("gaussian_*.npz"))
        assert cache_files_after_params == cache_files_after_yaml


def test_splits_from_arrays_builds_three_nonempty_splits() -> None:
    n = 200
    z = np.random.default_rng(0).normal(size=(2 * n, 1))
    x = np.random.default_rng(1).normal(size=(2 * n, 1))
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])

    splits = RANDataset(batch_size=32).splits_from_arrays(z, x, y)

    for ds in (splits.train, splits.val, splits.test):
        features, labels = next(iter(ds))
        assert set(features.keys()) == {"z", "x"}
        assert features["z"].shape[-1] == 1
        assert labels.shape[0] > 0


def _toy_splits(n: int = 200, batch_size: int = 32, **kwargs) -> DatasetSplits:
    z = np.arange(2 * n, dtype=np.double).reshape(-1, 1)
    x = -z
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    return RANDataset(batch_size=batch_size, **kwargs).splits_from_arrays(z, x, y)


class TestArrayDataset:
    """Behaviour the tf.data pipeline used to provide."""

    def test_splits_partition_events_without_overlap(self) -> None:
        """Every event lands in exactly one split."""
        splits = _toy_splits(n=200)
        ids = [set(ds.as_arrays()[0].ravel().tolist()) for ds in splits]
        assert sum(len(s) for s in ids) == 400
        assert set.union(*ids) == set(range(400))
        assert not (ids[0] & ids[1])
        assert not (ids[0] & ids[2])
        assert not (ids[1] & ids[2])

    def test_default_split_fractions(self) -> None:
        splits = _toy_splits(n=500)
        assert splits.test.n_events == 200  # 20% of 1000
        assert splits.val.n_events == 100  # 10% of 1000
        assert splits.train.n_events == 700

    def test_shuffle_interleaves_classes(self) -> None:
        """Splits must not be single-class: data and MC arrive stacked."""
        for ds in _toy_splits(n=500):
            frac = float(ds.as_arrays()[2].mean())
            assert 0.4 < frac < 0.6, f"class fraction {frac} — split is not mixed"

    def test_z_and_x_stay_paired(self) -> None:
        """Shuffling and splitting must not decouple particle/detector rows."""
        for ds in _toy_splits(n=200):
            z, x, _ = ds.as_arrays()
            np.testing.assert_array_equal(x, -z)
            for features, _ in ds:
                np.testing.assert_array_equal(features["x"], -features["z"])

    def test_batches_cover_split_and_keep_remainder(self) -> None:
        """A short final batch is kept, not dropped."""
        splits = _toy_splits(n=200, batch_size=32)
        train = splits.train
        assert train.n_events % 32 != 0, "test needs a ragged final batch"
        sizes = [len(y) for _, y in train]
        assert len(sizes) == len(train)
        assert sum(sizes) == train.n_events
        assert all(s == 32 for s in sizes[:-1])
        assert sizes[-1] == train.n_events % 32

    def test_train_reshuffles_each_epoch_but_val_does_not(self) -> None:
        splits = _toy_splits(n=200)

        def first(ds: ArrayDataset) -> ndarray[tuple[int], dtype[float64]]:
            return next(iter(ds))[0]["z"].ravel().copy()

        assert not np.array_equal(first(splits.train), first(splits.train))
        np.testing.assert_array_equal(first(splits.val), first(splits.val))

    def test_reset_rewinds_the_shuffle_sequence(self) -> None:
        """Two passes after a reset must repeat the first two exactly.

        Without this, a second training run over the same splits would resume
        mid-sequence and see different data — silently coupling the dataset
        seed to how many runs preceded it.
        """
        train = _toy_splits(n=200).train

        def firsts():
            return [next(iter(train))[0]["z"].ravel().copy() for _ in range(2)]

        before = firsts()
        assert not np.array_equal(before[0], before[1]), "passes should differ"
        train.reset()
        for a, b in zip(before, firsts(), strict=False):
            np.testing.assert_array_equal(a, b)

    def test_shuffle_order_is_independent_of_other_iterations(self) -> None:
        """Pass N's order depends only on (seed, N), not on who else iterated."""
        a, b = _toy_splits(n=200).train, _toy_splits(n=200).train
        for _ in range(3):  # advance `a` only
            list(a)
        a.reset()
        np.testing.assert_array_equal(
            next(iter(a))[0]["z"].ravel(), next(iter(b))[0]["z"].ravel()
        )

    def test_as_arrays_matches_iteration_for_ordered_splits(self) -> None:
        val = _toy_splits(n=200).val
        z_iter = np.concatenate([f["z"] for f, _ in val], axis=0)
        np.testing.assert_array_equal(z_iter, val.as_arrays()[0])

    def test_same_seed_gives_same_split(self) -> None:
        a, b = _toy_splits(n=200, seed=7), _toy_splits(n=200, seed=7)
        np.testing.assert_array_equal(a.test.as_arrays()[0], b.test.as_arrays()[0])

    def test_mismatched_lengths_raise(self) -> None:
        from ran.data.datasets import ArrayDataset

        with pytest.raises(ValueError, match="first dimension"):
            ArrayDataset(
                np.zeros((5, 1)), np.zeros((4, 1)), np.zeros(5, dtype=np.ubyte)
            )

    def test_too_few_events_to_split_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one event"):
            _toy_splits(n=2)
