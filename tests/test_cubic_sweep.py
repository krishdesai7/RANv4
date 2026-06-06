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


class _FakeTensor:
    """Minimal stand-in for a Keras tensor: only needs .numpy()."""

    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def test_run_point_wiring_with_stubbed_training(tmp_path, monkeypatch):
    """Verify run_point's orchestration WITHOUT training any models.

    Real RAN/OmniFold training is cluster work; here we stub train() and
    omnifold_unfold() with instant fakes and only check that run_point draws
    the right s, normalizes weights, computes both metrics, and writes the JSON.
    """
    import ran.experiments.cubic_sweep as cs

    def fake_train(splits, dim=1, n_epochs=100):
        # Generator returns uniform raw weights for any z (shape (n, 1)).
        def g(z):
            return _FakeTensor(np.ones((len(z), 1)))

        return g, None, None

    def fake_omnifold(x_data, x_sim, z_gen, niter=3, epochs=50, batch_size=512):
        return np.ones(len(z_gen))

    monkeypatch.setattr(cs, "train", fake_train)
    monkeypatch.setattr(cs, "omnifold_unfold", fake_omnifold)

    out = cs.run_point(s_index=3, sweep_dir=tmp_path, n_samples=2000, n_points=25, seed=0)

    assert out["s_index"] == 3
    assert out["s"] == float(np.linspace(0.0, 20.0, 25)[3])
    assert np.isfinite(out["ran_wd"])
    assert np.isfinite(out["omnifold_wd"])

    written = json.loads((tmp_path / "s_03.json").read_text())
    assert written == out


def test_collect_writes_results_and_plot(tmp_path):
    from ran.experiments.cubic_sweep import collect

    for i, s in enumerate([0.0, 10.0]):
        rec = {"s_index": i, "s": s, "ran_wd": 0.1 * (i + 1), "omnifold_wd": 0.2 * (i + 1)}
        (tmp_path / f"s_{i:02d}.json").write_text(json.dumps(rec))

    collect(sweep_dir=tmp_path, n_points=2)

    assert (tmp_path / "results.npz").exists()
    assert (tmp_path / "wasserstein_vs_s.pdf").exists()

    data = np.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(data["s"], [0.0, 10.0])
    np.testing.assert_allclose(data["ran"], [0.1, 0.2])
    np.testing.assert_allclose(data["omnifold"], [0.2, 0.4])
