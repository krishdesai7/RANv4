import numpy as np
from ran.evaluate import (
    _js_per_dim,
    _normalized_histograms,
    _triangular_per_dim,
)


def test_normalized_histograms_treats_1d_samples_as_one_feature() -> None:
    p, q = _normalized_histograms(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
        n_bins=2,
    )

    assert p.shape == (1, 2)
    assert q.shape == (1, 2)
    np.testing.assert_allclose(p, [[0.5, 0.5]])
    np.testing.assert_allclose(q, [[0.25, 0.75]])


def test_normalized_histograms_weights_only_comparison_samples() -> None:
    ref = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    comp = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    p, q = _normalized_histograms(
        ref,
        comp,
        weights=np.array([1.0, 1.0, 2.0]),
        n_bins=2,
    )

    np.testing.assert_allclose(p, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(q, [[0.25, 0.75], [0.5, 0.5]])


def test_js_per_dim_reduces_over_histogram_bins() -> None:
    ref = np.array([[0.0, 0.0], [0.0, 1.0]])
    comp = np.array([[1.0, 0.0], [1.0, 1.0]])

    result = _js_per_dim(ref, comp, n_bins=2)

    np.testing.assert_allclose(result, [np.log(2.0), 0.0], atol=1e-15)


def test_triangular_per_dim_reduces_over_histogram_bins() -> None:
    ref = np.array([[0.0, 0.0], [0.0, 1.0]])
    comp = np.array([[1.0, 0.0], [1.0, 1.0]])

    result = _triangular_per_dim(ref, comp, n_bins=2)

    np.testing.assert_allclose(result, [2000.0, 0.0], atol=1e-12)
