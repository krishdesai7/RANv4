import numpy as np
import pytest
from pathlib import Path
import yaml
from ran.data.config import parse_gaussian_config, sigma_to_covariance

class TestSigmaToCovariance:
    """Test the three sigma forms: scalar, vector, matrix."""

    def test_scalar_1d(self):
        cov = sigma_to_covariance(2.0, dim=1)
        expected = np.array([[4.0]])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_scalar_3d(self):
        cov = sigma_to_covariance(1.5, dim=3)
        expected = 2.25 * np.eye(3)
        np.testing.assert_array_almost_equal(cov, expected)

    def test_vector(self):
        cov = sigma_to_covariance([1.0, 2.0], dim=2)
        expected = np.diag([1.0, 4.0])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_matrix_passthrough(self):
        mat = [[1.0, 0.5], [0.5, 2.0]]
        cov = sigma_to_covariance(mat, dim=2)
        np.testing.assert_array_almost_equal(cov, mat)

    def test_vector_wrong_dim(self):
        with pytest.raises(ValueError, match="dim"):
            sigma_to_covariance([1.0, 2.0, 3.0], dim=2)

    def test_matrix_wrong_shape(self):
        with pytest.raises(ValueError, match="dim"):
            sigma_to_covariance([[1.0, 0.0], [0.0, 1.0]], dim=3)

    def test_not_positive_definite(self):
        """A matrix with negative eigenvalue should fail."""
        bad = [[1.0, 5.0], [5.0, 1.0]]
        with pytest.raises(np.linalg.LinAlgError):
            sigma_to_covariance(bad, dim=2)

    def test_asymmetric_matrix_raises(self):
        """An asymmetric matrix should be rejected."""
        asym = [[1.0, 0.5], [999.0, 2.0]]
        with pytest.raises(ValueError, match="symmetric"):
            sigma_to_covariance(asym, dim=2)

    def test_negative_scalar_raises(self):
        """Negative scalar sigma is physically nonsensical."""
        with pytest.raises(ValueError, match="negative"):
            sigma_to_covariance(-1.0, dim=2)

    def test_negative_vector_element_raises(self):
        """Negative elements in sigma vector should be rejected."""
        with pytest.raises(ValueError, match="negative"):
            sigma_to_covariance([1.0, -0.5], dim=2)


class TestParseGaussianConfig:
    """Test full YAML config parsing."""

    def _write_yaml(self, data: dict, tmp_path: Path) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        return p

    def test_valid_2d_config(self, tmp_path):
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = self._write_yaml(cfg, tmp_path)
        params = parse_gaussian_config(path)
        assert params["dim"] == 2
        assert params["mu_mc"].shape == (2,)
        assert params["cov_mc"].shape == (2, 2)
        assert params["cov_true"].shape == (2, 2)
        assert params["cov_detector"].shape == (2, 2)

    def test_scalar_sigma(self, tmp_path):
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        params = parse_gaussian_config(path)
        assert params["dim"] == 1
        np.testing.assert_array_almost_equal(params["cov_mc"], [[1.0]])
        np.testing.assert_array_almost_equal(params["cov_detector"], [[0.25]])

    def test_missing_key(self, tmp_path):
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="missing"):
            parse_gaussian_config(path)

    def test_dim_mismatch(self, tmp_path):
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="dim"):
            parse_gaussian_config(path)
