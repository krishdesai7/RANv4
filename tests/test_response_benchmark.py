from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from benchmarks.response import Pairs, PseudoDomainRule, main, response_statistic


def test_response_statistic_is_paired_held_out_bce_gain() -> None:
    statistic = response_statistic(
        labels=np.array([1.0, 1.0, 0.0, 0.0]),
        p_z=np.full(4, 0.5),
        p_zx=np.array([0.75, 0.75, 0.25, 0.25]),
    )

    assert math.isclose(statistic.bce_z, math.log(2.0))
    assert math.isclose(statistic.bce_zx, -math.log(0.75))
    assert math.isclose(statistic.delta_nats, math.log(1.5))
    assert math.isclose(statistic.delta_bits, math.log(1.5) / math.log(2.0))
    assert math.isclose(statistic.standard_error, 0.0, abs_tol=1e-15)


def test_response_statistic_uses_fixed_class_stratified_standard_error() -> None:
    statistic = response_statistic(
        labels=np.array([1.0, 1.0, 0.0, 0.0]),
        p_z=np.full(4, 0.5),
        p_zx=np.full(4, 0.75),
    )

    assert math.isclose(statistic.standard_error, 0.0, abs_tol=1e-15)


@pytest.mark.parametrize("bad_probability", [np.nan, -0.1, 1.1])
def test_response_statistic_rejects_invalid_probabilities(
    bad_probability: float,
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        response_statistic(
            labels=np.array([1.0, 1.0, 0.0, 0.0]),
            p_z=np.array([0.5, bad_probability, 0.5, 0.5]),
            p_zx=np.array([0.75, 0.75, 0.25, 0.25]),
        )


def test_pseudo_domain_rule_creates_reproducible_covariate_shift() -> None:
    z = np.linspace(-3.0, 3.0, 2_000, dtype=np.single)[:, None]
    x = np.column_stack((z[:, 0], np.square(z[:, 0]))).astype(np.single)
    pairs = Pairs(z=z, x=x)
    rule = PseudoDomainRule.fit(z, seed=7, strength=1.5)

    first = rule.partition(pairs, seed=11)
    second = rule.partition(pairs, seed=11)

    assert np.array_equal(first.positive.z, second.positive.z)
    assert np.array_equal(first.negative.x, second.negative.x)
    assert len(first.positive.z) > 500
    assert len(first.negative.z) > 500
    assert first.positive.z[:, 0].mean() > first.negative.z[:, 0].mean() + 1.0
    assert np.array_equal(first.positive.x[:, 0], first.positive.z[:, 0])


def test_pseudo_domain_assignment_does_not_depend_on_detector_values() -> None:
    z = np.linspace(-2.0, 2.0, 500, dtype=np.single)[:, None]
    original = Pairs(z=z, x=z.copy())
    altered = Pairs(z=z, x=np.flip(z, axis=0).copy())
    rule = PseudoDomainRule.fit(z, seed=3, strength=1.0)

    first = rule.partition(original, seed=5)
    second = rule.partition(altered, seed=5)

    assert np.array_equal(first.positive.z, second.positive.z)
    assert np.array_equal(first.negative.z, second.negative.z)


def test_cli_rejects_invalid_enabled_null_strength(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "response.py",
            "--null-repeats",
            "1",
            "--null-strength",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "--null-strength must be positive and finite" in capsys.readouterr().err
