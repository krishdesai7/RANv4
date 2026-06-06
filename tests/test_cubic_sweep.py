import json
import numpy as np
from scipy.stats import wasserstein_distance

from ran.experiments.cubic_sweep import (
    response,
    make_particles,
    unfolded_wasserstein,
)


def test_response_identity_at_zero():
    z = np.linspace(-3, 3, 100)
    np.testing.assert_array_equal(response(0.0, z), z)


def test_response_monotonic_for_positive_s():
    z = np.linspace(-3, 3, 1000)
    out = response(5.0, z)
    assert np.all(np.diff(out) > 0)


def test_make_particles_shapes_and_means():
    z_truth, z_gen = make_particles(50_000, seed=123)
    assert z_truth.shape == (50_000,)
    assert z_gen.shape == (50_000,)
    assert abs(z_truth.mean() - 0.0) < 0.05
    assert abs(z_gen.mean() - (-1.0)) < 0.05


def test_unfolded_wasserstein_uniform_weights_equals_unweighted():
    rng = np.random.default_rng(0)
    z_truth = rng.normal(0, 1, 5000)
    z_gen = rng.normal(-1, 1, 5000)
    w = np.ones_like(z_gen)
    got = unfolded_wasserstein(z_truth, z_gen, w)
    expected = wasserstein_distance(z_truth, z_gen)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
