"""Tests for the MMD resolution floor measurement.

The floor decides which hyperparameter arms are *admissible* -- whether an arm
is one the selection criterion can distinguish from the best, and so one a
truth-free pipeline would refuse to ship. Until now the number in use, 2.5e-4,
was an extrapolation: `benchmarks/README.md` measures ~5e-4 at m=8192 and scales
it by 1/m to the operating point of `MMD_SUBSAMPLE = 16384`. That extrapolation
is load-bearing for every admissibility call in the dispersion sweep, so it has
to become a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.mmd_floor import FloorEstimate, null_floor


def _one_population(n: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.single)


class TestNullFloor:
    def test_the_null_estimate_is_consistent_with_zero(self) -> None:
        """Both halves come from one population, so the true MMD^2 is exactly 0.

        Splitting one sample is what makes this a null by construction rather
        than by assumption -- no appeal to the two generators being close.
        """
        estimate = null_floor(_one_population(4096), m=512, repeats=16, seed=0)

        assert abs(estimate.mean) < 3.0 * estimate.standard_error

    def test_the_estimator_takes_both_signs(self) -> None:
        """The unbiased U-statistic is not a distance and goes negative.

        A floor read off the magnitudes alone would be half the truth: the
        spread is two-sided, which is why an arm at -2e-4 and one at +2e-4 are
        equally indistinguishable from a perfect match.
        """
        estimate = null_floor(_one_population(4096), m=512, repeats=16, seed=0)

        assert min(estimate.values) < 0.0 < max(estimate.values)

    def test_the_spread_shrinks_as_the_subsample_grows(self) -> None:
        """The 1/m scaling the old extrapolation assumed, actually checked."""
        small = null_floor(_one_population(8192), m=256, repeats=24, seed=1)
        large = null_floor(_one_population(8192), m=1024, repeats=24, seed=1)

        assert large.sd < small.sd

    def test_it_is_reproducible_from_its_seed(self) -> None:
        first = null_floor(_one_population(2048), m=256, repeats=8, seed=3)
        second = null_floor(_one_population(2048), m=256, repeats=8, seed=3)

        assert first.values == pytest.approx(second.values)

    def test_repeats_draw_different_subsamples(self) -> None:
        """Otherwise the 'spread' would be zero and the floor would read as 0."""
        estimate = null_floor(_one_population(2048), m=256, repeats=8, seed=5)

        assert len(set(estimate.values)) == len(estimate.values)

    def test_it_refuses_a_sample_too_small_to_split(self) -> None:
        with pytest.raises(ValueError, match="two disjoint"):
            null_floor(_one_population(100), m=256, repeats=4, seed=0)


class TestFloorEstimate:
    def test_reports_the_standard_error_of_the_mean_not_the_spread(self) -> None:
        """Two different questions, and the admissibility one wants the spread.

        `sd` is how far a single run's number wanders, which is what decides
        whether two arms differ. `standard_error` only says how well this
        benchmark has pinned the mean.
        """
        estimate = FloorEstimate(values=(1.0, 2.0, 3.0, 4.0))

        assert estimate.mean == pytest.approx(2.5)
        assert estimate.sd == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
        assert estimate.standard_error == pytest.approx(estimate.sd / 2.0)


class TestPoolOverlap:
    """Repeats drawn from too small a pool share most of their rows.

    Measured on 12-dimensional Gaussians, the null SD scales as the theoretical
    `m^-1` while `2m/N` stays small (exponents -1.00 and -0.93 at 2m/N <= 0.2),
    but flattens to `m^-0.74` once the draws reach 2m/N = 0.4 -- successive
    repeats are then largely the same rows, and the spread they show is not the
    estimator's. A floor measured that way comes out biased and the extrapolated
    value it is checked against would look wrong for the wrong reason.
    """

    def test_warns_when_the_draws_cover_too_much_of_the_pool(self, caplog) -> None:
        x = _one_population(1000)

        with caplog.at_level("WARNING"):
            null_floor(x, m=400, repeats=4, seed=0)

        assert "2m/N" in caplog.text

    def test_stays_quiet_when_the_pool_is_ample(self, caplog) -> None:
        x = _one_population(20_000)

        with caplog.at_level("WARNING"):
            null_floor(x, m=400, repeats=4, seed=0)

        assert "2m/N" not in caplog.text


def test_fitted_exponent_recovers_a_power_law() -> None:
    """The report states the measured scaling rather than assuming 1/m."""
    from benchmarks.mmd_floor import fitted_exponent

    exponent = fitted_exponent({1000: 1e-3, 2000: 5e-4, 4000: 2.5e-4})

    assert exponent == pytest.approx(-1.0, abs=1e-6)


class TestFloorUncertainty:
    """The floor is an estimate, and near a boundary its own error matters.

    The dispersion arms sit 0-1 floors from the tie threshold, so a floor
    quoted without an error bar invites reading a 1.1 as meaningfully different
    from a 0.9. For a normal sample the standard error of an SD is
    `sd / sqrt(2(n-1))`, which is ~9% at the 64 repeats the benchmark defaults
    to and ~13% at 32.
    """

    def test_reports_the_standard_error_of_the_spread_itself(self) -> None:
        estimate = FloorEstimate(values=tuple(float(v) for v in range(65)))

        expected = estimate.sd / np.sqrt(2.0 * (len(estimate.values) - 1))

        assert estimate.sd_error == pytest.approx(expected)

    def test_more_repeats_pin_the_floor_more_tightly(self) -> None:
        rng = np.random.default_rng(0)
        few = FloorEstimate(values=tuple(rng.normal(size=16)))
        many = FloorEstimate(values=tuple(rng.normal(size=256)))

        assert many.sd_error / many.sd < few.sd_error / few.sd
