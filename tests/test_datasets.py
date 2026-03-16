import numpy as np
import pytest
from pathlib import Path

import yaml

from ran.data.datasets import RAN_Dataset


def _write_config(params: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(params))
    return p


class TestGenerateGaussianDataset:
    """Test multivariate Gaussian dataset generation."""

    def test_1d_uncorrelated(self, tmp_path):
        """1D scalar sigma should produce valid splits."""
        cfg = {
            "mu_mc": [0.5],
            "mu_true": [0.0],
            "sigma_mc": 0.9,
            "sigma_true": 1.0,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=1000)
        assert splits.train is not None
        assert splits.val is not None
        assert splits.test is not None

    def test_2d_correlated_shapes(self, tmp_path):
        """2D with full covariance should produce correct shapes."""
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [[1.0, -0.54], [-0.54, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=2000)
        for features, y in splits.test:
            assert features["z"].shape[1] == 2
            assert features["x"].shape[1] == 2
            break

    def test_params_dict_interface(self, tmp_path):
        """Passing params dict directly should work (for --load_run)."""
        params = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(params=params, n_samples=1000)
        assert splits.train is not None

    def test_both_config_and_params_raises(self, tmp_path):
        """Providing both config_path and params should error."""
        cfg = {"mu_mc": [0.0], "mu_true": [0.5], "sigma_mc": 1.0,
               "sigma_true": 0.9, "sigma_detector": 0.5}
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(config_path=path, params=cfg, n_samples=100)

    def test_neither_config_nor_params_raises(self):
        """Providing neither config_path nor params should error."""
        ds = RAN_Dataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(n_samples=100)

    def test_caching(self, tmp_path):
        """Second call with same config should hit cache."""
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"
        ds = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files = list(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files) == 1
        ds2 = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds2.generate_gaussian_dataset(config_path=path, n_samples=500)

    def test_smearing_preserves_event_coupling(self, tmp_path):
        """Detector-level values should be correlated with particle-level."""
        cfg = {
            "mu_mc": [0.0, 0.0],
            "mu_true": [0.0, 0.0],
            "sigma_mc": [1.0, 1.0],
            "sigma_true": [1.0, 1.0],
            "sigma_detector": [0.1, 0.1],
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=10000, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=10000)
        for features, y in splits.test:
            z = features["z"].numpy()
            x = features["x"].numpy()
            for d in range(2):
                corr = np.corrcoef(z[:, d], x[:, d])[0, 1]
                assert corr > 0.95, f"dim {d}: corr={corr}, expected >0.95"
            break

    def test_yaml_and_params_share_cache(self, tmp_path):
        """YAML path and equivalent params dict must produce the same cache key."""
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"

        ds1 = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds1.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files_after_yaml = set(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files_after_yaml) == 1

        reload_params = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [[1.0, 0.0], [0.0, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [[0.25, 0.0], [0.0, 0.64]],
        }
        ds2 = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds2.generate_gaussian_dataset(params=reload_params, n_samples=500)

        cache_files_after_params = set(cache_dir.glob("gaussian_*.npz"))
        assert cache_files_after_params == cache_files_after_yaml
