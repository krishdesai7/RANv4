from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import ran.experiments.cubic_sweep as cs
import ran.train
from ran.experiments.cubic_sweep import (
    _sweep_point,
    collect,
    make_particles,
    response,
    unfolded_wasserstein,
)
from ran.train import TrainResult
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from ran.rantypes import RANModel


def test_response_identity_at_zero() -> None:
    z = np.linspace(-3, 3, 100)
    (out,) = response(np.double(0.0), z)
    np.testing.assert_array_equal(out, z)


def test_response_monotonic_for_positive_s() -> None:
    z = np.linspace(-3, 3, 1000)
    (out,) = response(np.double(5.0), z)
    assert np.all(np.diff(out) > 0)


def test_response_maps_each_sample_in_order() -> None:
    """One array in, one array out, however many are passed at once."""
    z1, z2 = np.linspace(-3, 3, 100), np.linspace(-1, 1, 100)
    both = response(np.double(2.0), z1, z2)
    assert len(both) == 2
    for got, z in zip(both, (z1, z2), strict=True):
        np.testing.assert_array_equal(got, z + 2.0 * z**3)


def test_make_particles_shapes_and_means() -> None:
    z_truth, z_gen = make_particles(50_000, seed=123)
    assert z_truth.shape == (50_000,)
    assert z_gen.shape == (50_000,)
    assert abs(z_truth.mean() - 0.0) < 0.05
    assert abs(z_gen.mean() - (-1.0)) < 0.05


def test_unfolded_wasserstein_uniform_weights_equals_unweighted() -> None:
    rng = np.random.default_rng(0)
    z_truth = rng.normal(0, 1, 5000)
    z_gen = rng.normal(-1, 1, 5000)
    w = np.ones_like(z_gen)
    got = unfolded_wasserstein(z_truth, z_gen, w)
    expected = wasserstein_distance(z_truth, z_gen)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_run_ran_wiring_with_stubbed_training(tmp_path, monkeypatch) -> None:
    """Verify run_ran's orchestration WITHOUT training any models.

    Real RAN training is cluster work; here train() is stubbed with an instant
    fake and we only check that run_ran draws the right s, normalizes weights,
    computes the metric, and writes the JSON.
    """

    def fake_train(_splits, *, seed=None, **_kwargs) -> TrainResult:
        # Generator returns uniform raw weights for any z (shape (n, 1)).
        # dim/n_epochs are swallowed by **_kwargs: this stub only cares that
        # run_ran passes a seed through.
        return TrainResult(
            # TrainResult declares g/d as RANModel; run_ran only calls g and
            # reads seed, so this test double is sufficient at runtime.
            g=cast("RANModel", lambda z: np.ones((len(z), 1))),
            d=cast("RANModel", None),
            history={},
            seed=seed or 0,
        )

    # run_ran imports train() lazily from ran.train, so patch it at the source.
    monkeypatch.setattr(ran.train, "train", fake_train)

    out = cs.run_ran(
        s_index=3, sweep_dir=tmp_path, n_samples=2000, n_points=25, seed=0, init_seed=5
    )

    assert out["s_index"] == 3
    assert out["s"] == pytest.approx(float(np.linspace(0.0, 20.0, 25)[3]))
    assert np.isfinite(out["ran_wd"])
    # Both seeds recorded, so the point can be reproduced from its own JSON.
    assert out["seed"] == 0
    assert out["init_seed"] == 5

    assert json.loads((tmp_path / "ran_03.json").read_text()) == out


def test_sweep_point_returns_columns_with_truth() -> None:
    """Both unfolders take (n, 1); the answer key is real, not the sentinel."""
    _, pops = _sweep_point(s_index=4, n_points=25, n_samples=1000, seed=0)
    for a in (pops.mc.z, pops.mc.x, pops.data, pops.truth):
        assert a.shape == (1000, 1)
    assert pops.has_truth


def _write_points(
    tmp_path, indices: list[int], s_values: list[float], ran=True
) -> None:
    for i, s in zip(indices, s_values, strict=False):
        if ran:
            (tmp_path / f"ran_{i:02d}.json").write_text(
                json.dumps({"s_index": i, "s": s, "ran_wd": 0.1 * (i + 1)})
            )


def test_collect_joins_both_methods_and_writes_results_and_plot(tmp_path) -> None:

    _write_points(tmp_path, [0, 1], [0.0, 10.0])
    collect(sweep_dir=tmp_path, n_points=2)

    assert (tmp_path / "results.npz").exists()
    assert (tmp_path / "wasserstein_vs_s.pdf").exists()

    data = np.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(data["s"], [0.0, 10.0])
    np.testing.assert_allclose(data["ran"], [0.1, 0.2])
