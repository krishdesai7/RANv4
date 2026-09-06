from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from ran.data import (
    gaussian_config_from_run_config,
    parse_gaussian_config,
    sigma_to_covariance,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ran.rantypes import GaussianConfig


class TestSigmaToCovariance:
    """Test the three sigma forms: scalar, vector, matrix."""

    def test_scalar_1d(self) -> None:
        cov = sigma_to_covariance(2.0, 1)
        expected = np.array([[4.0]])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_scalar_3d(self) -> None:
        cov = sigma_to_covariance(1.5, 3)
        expected = 2.25 * np.eye(3)
        np.testing.assert_array_almost_equal(cov, expected)

    def test_vector(self) -> None:
        cov = sigma_to_covariance([1.0, 2.0], 2)
        expected = np.diag([1.0, 4.0])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_matrix_passthrough(self) -> None:
        mat = [[1.0, 0.5], [0.5, 2.0]]
        cov = sigma_to_covariance(mat, 2)
        np.testing.assert_array_almost_equal(cov, mat)

    def test_vector_wrong_dim(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            _ = sigma_to_covariance([1.0, 2.0, 3.0], 2)

    def test_matrix_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            _ = sigma_to_covariance([[1.0, 0.0], [0.0, 1.0]], 3)

    def test_not_positive_definite(self) -> None:
        """A matrix with negative eigenvalue should fail."""
        bad = [[1.0, 5.0], [5.0, 1.0]]
        with pytest.raises(np.linalg.LinAlgError):
            _ = sigma_to_covariance(bad, 2)

    def test_asymmetric_matrix_raises(self) -> None:
        """An asymmetric matrix should be rejected."""
        asym = [[1.0, 0.5], [999.0, 2.0]]
        with pytest.raises(ValueError, match="symmetric"):
            _ = sigma_to_covariance(asym, 2)

    def test_negative_scalar_raises(self) -> None:
        """Negative scalar sigma is physically nonsensical."""
        with pytest.raises(ValueError, match="negative"):
            _ = sigma_to_covariance(-1.0, 2)

    def test_negative_vector_element_raises(self) -> None:
        """Negative elements in sigma vector should be rejected."""
        with pytest.raises(ValueError, match="negative"):
            _ = sigma_to_covariance([1.0, -0.5], 2)


class TestParseGaussianConfig:
    """Test full YAML config parsing.

    These configs are written in the *input* format -- `mu_*` plus the
    `sigma_*` keys in any of the three accepted forms -- which is what a
    params/*.yaml file holds. `parse_gaussian_config` promotes them to the
    covariance matrices of a `GaussianConfig`; that is its output, not its
    input, so it never round-trips through `GaussianConfig.model_dump()`.
    """

    def _write_yaml(self, data: dict[str, Any], tmp_path: Path) -> Path:
        p: Path = tmp_path / "config.yaml"
        _ = p.write_text(yaml.dump(data))
        return p

    def test_valid_2d_config(self, tmp_path: Path) -> None:
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = self._write_yaml(cfg, tmp_path)
        params: GaussianConfig = parse_gaussian_config(path)
        assert params.dim == 2
        assert params.mu_gen.shape == (2,)
        assert params.cov_gen.shape == (2, 2)
        assert params.cov_true.shape == (2, 2)
        assert params.cov_detector.shape == (2, 2)

    def test_scalar_sigma(self, tmp_path: Path) -> None:
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        params: GaussianConfig = parse_gaussian_config(path)
        assert params.dim == 1
        np.testing.assert_array_almost_equal(params.cov_gen, [[1.0]])
        np.testing.assert_array_almost_equal(params.cov_detector, [[0.25]])

    def test_missing_key(self, tmp_path: Path) -> None:
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="missing"):
            _ = parse_gaussian_config(path)

    def test_dim_mismatch(self, tmp_path: Path) -> None:
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="dim"):
            _ = parse_gaussian_config(path)


class TestGaussianConfigFromRunConfig:
    """Reading the `gaussian_params` block back out of a run's config.json.

    Three formats exist in runs/ and two share their key names, so this is the
    one place that has to get the disambiguation right. A `--load-run` replot
    and `ran evaluate` both go through here; when they read it independently
    they drifted, and metrics on every Gaussian run died with KeyError.
    """

    def test_current_format_uses_covariances_as_written(self) -> None:
        params = gaussian_config_from_run_config(
            {
                "dim": 2,
                "mu_gen": [0.0, 1.0],
                "mu_true": [0.2, 0.8],
                "cov_gen": [[1.0, 0.5], [0.5, 2.25]],
                "cov_true": [[0.81, -0.5], [-0.5, 1.69]],
                "cov_detector": [[0.25, 0.0], [0.0, 0.64]],
            },
            dim=2,
        )
        assert params.dim == 2
        np.testing.assert_array_almost_equal(params.cov_gen, [[1.0, 0.5], [0.5, 2.25]])
        np.testing.assert_array_almost_equal(
            params.cov_detector, [[0.25, 0.0], [0.0, 0.64]]
        )

    def test_master_era_sigma_keys_holding_a_covariance_pass_through(self) -> None:
        """master's __main__ stored cov_* under the name sigma_*.

        A full matrix must survive unchanged -- squaring it again would quietly
        change the dataset every reloaded run regenerates.
        """
        params = gaussian_config_from_run_config(
            {
                "mu_gen": [0.0, 1.0],
                "mu_true": [0.2, 0.8],
                "sigma_gen": [[1.0, 0.5], [0.5, 2.25]],
                "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
                "sigma_detector": [[0.25, 0.0], [0.0, 0.64]],
            },
            dim=2,
        )
        np.testing.assert_array_almost_equal(params.cov_gen, [[1.0, 0.5], [0.5, 2.25]])

    def test_ancient_scalar_sigma_is_promoted(self) -> None:
        """The oldest runs stored a raw sigma, which still needs squaring."""
        params = gaussian_config_from_run_config(
            {
                "mu_gen": [0.5] * 4,
                "mu_true": [0.0] * 4,
                "sigma_gen": 0.9,
                "sigma_true": 1.0,
                "sigma_detector": 0.5,
            },
            dim=4,
        )
        np.testing.assert_array_almost_equal(params.cov_gen, 0.81 * np.eye(4))
        np.testing.assert_array_almost_equal(params.cov_detector, 0.25 * np.eye(4))

    def test_round_trips_through_model_dump(self) -> None:
        """What `_save_run` writes must be what this reads back."""
        source = {
            "dim": 2,
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "cov_gen": [[1.0, 0.5], [0.5, 2.25]],
            "cov_true": [[0.81, -0.5], [-0.5, 1.69]],
            "cov_detector": [[0.25, 0.0], [0.0, 0.64]],
        }
        first = gaussian_config_from_run_config(source, dim=2)
        second = gaussian_config_from_run_config(first.model_dump(), dim=2)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_almost_equal(np.asarray(a), np.asarray(b))

    def test_missing_covariance_key_is_named(self) -> None:
        with pytest.raises(ValueError, match="cov_detector"):
            _ = gaussian_config_from_run_config(
                {
                    "mu_gen": [0.0],
                    "mu_true": [0.5],
                    "sigma_gen": 1.0,
                    "sigma_true": 0.9,
                },
                dim=1,
            )

    def test_missing_mu_is_reported_as_missing(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            _ = gaussian_config_from_run_config({"sigma_gen": 1.0}, dim=1)
