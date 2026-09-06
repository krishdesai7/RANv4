from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from ran.baselines import _shared as shared
from ran.baselines import ibu
from ran.data import ArrayDataset
from ran.rantypes import ZXY, DatasetSplits, Events

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from ran.rantypes.events import Populations


def _split(z: list[list[float]], x: list[list[float]], y: list[int]) -> ArrayDataset:
    return ArrayDataset(
        ZXY(
            Events(np.asarray(z, dtype=np.single), np.asarray(x, dtype=np.single)),
            np.asarray(y, dtype=np.ubyte),
        ),
        batch_size=8,
    )


def _splits() -> DatasetSplits:
    return DatasetSplits(
        train=_split(
            [[0.2], [0.4], [1.2], [1.4]],
            [[-1.0], [0.5], [1.1], [3.0]],
            [0, 1, 0, 1],
        ),
        val=_split([[0.6], [1.6]], [[0.7], [1.7]], [0, 1]),
        test=_split(
            [[0.8], [1.8], [0.9], [1.9]],
            [[0.9], [2.5], [0.8], [2.2]],
            [0, 0, 1, 1],
        ),
    )


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset": "gaussian",
        "dim": 2,
        "n_samples": 100,
        "batch_size": 16,
        "data_seed": 7,
    }
    config.update(overrides)
    return config


def test_parse_config_builds_gaussian_variable_names() -> None:
    parsed = shared.parse_run_config(_config())

    assert parsed.dataset == "gaussian"
    assert parsed.variable_names == ("dim_0", "dim_1")
    assert parsed.dim == 2
    assert parsed.data_seed == 7


def test_parse_config_requires_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _ = shared.parse_run_config(["not", "a", "mapping"])


def test_parse_config_validates_jet_variable_count() -> None:
    with pytest.raises(ValueError, match=r"variables.*dim"):
        _ = shared.parse_run_config(_config(dataset="jets", variables=["mass"], dim=2))


@pytest.mark.parametrize("key", ["dim", "n_samples", "batch_size"])
def test_parse_config_requires_positive_integer_fields(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _ = shared.parse_run_config(_config(**{key: 0}))


def test_parse_config_reports_missing_required_field() -> None:
    config = _config()
    del config["n_samples"]

    with pytest.raises(ValueError, match="n_samples"):
        _ = shared.parse_run_config(config)


def test_parse_config_rejects_non_integer_data_seed() -> None:
    with pytest.raises(TypeError, match="data_seed"):
        _ = shared.parse_run_config(_config(data_seed="seven"))


def test_parse_config_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _ = shared.parse_run_config(_config(dataset="other"))


def test_parse_config_rejects_non_string_jet_variable() -> None:
    with pytest.raises(ValueError, match=r"variables.*strings"):
        _ = shared.parse_run_config(
            _config(dataset="jets", variables=["mass", 4], dim=2)
        )


def test_assign_bins_saturates_underflow_and_overflow() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.single)
    values = np.array([-3.0, 0.0, 0.4, 1.0, 2.0, 8.0], dtype=np.single)

    indices = ibu._assign_bins(values, edges)

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(ibu._bin_counts(indices, 2), [3.0, 3.0])


def test_saturating_counts_conserve_observed_population() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.single)
    observed = np.array([-2.0, 0.5, 4.0], dtype=np.single)

    counts = ibu._bin_counts(ibu._assign_bins(observed, edges), 2)

    assert counts.sum() == observed.size
    np.testing.assert_array_equal(counts, [2.0, 1.0])


def test_prepare_data_names_response_and_test_populations() -> None:
    fit, test = shared.prepare_populations(_splits(), expected_dim=1)

    # Train (2 MC + 2 nature) plus val (1 + 1) --- the test split's 2 + 2 are
    # excluded, which is the whole point of the field being called `fit`.
    assert fit.mc.z.shape == (3, 1)
    assert fit.mc.x.shape == (3, 1)
    assert fit.data.shape == (3, 1)
    assert test.mc.z.shape == (2, 1)
    assert test.data.shape == (2, 1)


def test_the_fit_population_contains_no_test_event() -> None:
    """The claim the metrics rest on, checked by value rather than by count.

    Fitting on every split and scoring a subset of it is the convention in the
    unfolding literature; it is also how this code behaved until the populations
    were split in two, so it is worth a test that fails loudly if `Split.ALL`
    ever comes back.
    """
    splits = _splits()
    fit, test = shared.prepare_populations(splits, expected_dim=1)

    def z_values(pops: Populations) -> set[float]:
        both = np.concatenate([pops.mc.z.ravel(), pops.data.ravel()])
        return {float(v) for v in both}

    fitted, held_out = z_values(fit), z_values(test)

    assert fitted.isdisjoint(held_out)
    # And accounts for every row: fit is exactly train+val, test exactly test.
    assert len(fit.mc) + len(fit.data) == splits.train.size + splits.val.size
    assert len(test.mc) + len(test.data) == splits.test.size


def test_prepare_data_rejects_configured_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="expected dim=2"):
        _ = shared.prepare_populations(_splits(), expected_dim=2)


def test_prepare_data_rejects_nonfinite_values() -> None:
    splits = _splits()
    splits.train.data.x[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _ = shared.prepare_populations(splits, expected_dim=1)


def test_prepare_data_rejects_empty_test_data_population() -> None:
    splits = _splits()
    without_test_data = DatasetSplits(
        train=splits.train,
        val=splits.val,
        test=_split([[0.8], [1.8]], [[0.9], [2.5]], [0, 0]),
    )

    with pytest.raises(ValueError, match="test split: populations must be nonempty"):
        _ = shared.prepare_populations(without_test_data, expected_dim=1)


def test_unfolded_to_bin_weights_uses_explicit_zero_prior_semantics() -> None:
    weights = ibu._unfolded_to_bin_weights(
        np.array([4.0, 0.0], dtype=np.single),
        np.array([2.0, 0.0], dtype=np.single),
    )

    np.testing.assert_array_equal(weights, [2.0, 0.0])


def test_unfolded_to_bin_weights_rejects_mass_without_prior() -> None:
    with pytest.raises(ValueError, match="zero-prior"):
        _ = ibu._unfolded_to_bin_weights(
            np.array([4.0, 1.0], dtype=np.single),
            np.array([2.0, 0.0], dtype=np.single),
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([0.0, 0.0]), "strictly positive"),
        (np.array([-1.0, 2.0]), "nonnegative"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, np.nan]), "finite"),
    ],
)
def test_normalize_weights_rejects_invalid_vectors(
    weights: NDArray[np.single], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ibu._normalize_weights(weights)


def test_normalize_weights_returns_mean_one() -> None:
    normalized = ibu._normalize_weights(np.array([1.0, 2.0, 3.0], dtype=np.single))

    assert normalized.mean() == pytest.approx(1.0)
    assert np.all(normalized >= 0)


def testunfold_variable_reports_insufficient_bins_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8], dtype=np.single),
        mc_sim=np.array([0.3, 0.7], dtype=np.single),
        observed=np.array([-1.0, 2.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.outcome.status == "skipped"
    assert result.outcome.skip_reason == "fewer than two purity bins"
    assert result.reweighting is None
    np.testing.assert_array_equal(
        result.weights_for(np.array([0.4, 0.6], dtype=np.single)), [1.0, 1.0]
    )


def testunfold_variable_returns_safe_mean_one_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
        mc_sim=np.array([-1.0, 0.7, 1.3, 3.0], dtype=np.single),
        observed=np.array([-2.0, 0.4, 1.4, 4.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )
    weights = result.weights_for(np.array([0.2, 1.8], dtype=np.single))

    assert result.outcome.status == "completed"
    assert result.outcome.n_bins == 2
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0)
    assert weights.mean() == pytest.approx(1.0)


def test_unfolds_in_single_precision_end_to_end() -> None:
    """IBU runs at float32 to match the precision its published results use.

    Real purity binning on a realistic number of events, so the population
    count checks and the mean-one postcondition are exercised against float32
    arithmetic rather than assumed to survive it.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    mc_gen = rng.normal(size=n).astype(np.single)
    mc_sim = (mc_gen + 0.3 * rng.normal(size=n).astype(np.single)).astype(np.single)
    observed = (0.5 + rng.normal(size=n).astype(np.single)).astype(np.single)

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=mc_gen,
        mc_sim=mc_sim,
        observed=observed,
        n_iterations=4,
        purity_threshold=0.5,
    )

    assert result.outcome.status == "completed"
    assert result.reweighting is not None
    assert result.reweighting.bin_weights.dtype == np.single
    assert result.reweighting.edges.dtype == np.single

    weights = result.weights_for(mc_gen)
    assert weights.dtype == np.single
    assert weights.mean(dtype=np.single) == pytest.approx(1.0, rel=1e-5)


def test_unfolding_applies_to_a_sample_it_was_not_fit_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The learned object is per-bin, so the sample it scores is a free choice."""
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
        mc_sim=np.array([-1.0, 0.7, 1.3, 3.0], dtype=np.single),
        observed=np.array([-2.0, 0.4, 1.4, 4.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.reweighting is not None
    assert result.reweighting.bin_weights.size == 2
    # Any number of events, in any order, so long as they are the same variable.
    for size in (2, 7, 50):
        gen = np.linspace(0.0, 2.0, size, dtype=np.single)
        assert result.weights_for(gen).shape == (size,)


@pytest.mark.parametrize(
    ("corrupt_call", "population"),
    [(1, "prior"), (2, "observed")],
)
def testunfold_variable_reports_population_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, corrupt_call: int, population: str
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )
    real_bin_counts: Callable[[NDArray[np.intp], int], NDArray[np.intp]] = (
        ibu._bin_counts
    )
    call_count = 0

    def dropping_bin_counts(indices: NDArray[np.intp], n_bins: int) -> NDArray[np.intp]:
        nonlocal call_count
        call_count += 1
        counts = real_bin_counts(indices, n_bins)
        if call_count == corrupt_call:
            # numpy stubs type a scalar element access as Any
            counts[-1] -= 1  # pyrefly: ignore[unknown-argument-type]
        return counts

    monkeypatch.setattr(ibu, "_bin_counts", dropping_bin_counts)

    with pytest.raises(ValueError, match=rf"{population}.*3.*4"):
        _ = ibu.unfold_variable(
            variable_name="dim_0",
            mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
            mc_sim=np.array([0.3, 0.7, 1.3, 1.7], dtype=np.single),
            observed=np.array([0.1, 0.9, 1.1, 1.9], dtype=np.single),
            n_iterations=2,
            purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
        )


def test_evaluate_dimension_accepts_one_dimensional_arrays() -> None:
    record = shared.evaluate_dimension(
        reference=np.array([0.0, 1.0, 2.0], dtype=np.single),
        comparison=np.array([0.0, 1.5, 3.0], dtype=np.single),
        weights=np.ones(3, dtype=np.single),
    )

    assert set(record) == {
        "wasserstein_before",
        "wasserstein_after",
        "wasserstein_improvement_pct",
        "jensenshannon_before",
        "jensenshannon_after",
        "jensenshannon_improvement_pct",
        "triangular_before",
        "triangular_after",
        "triangular_improvement_pct",
    }
    assert all(np.isfinite(cast("float", value)) for value in record.values())


def test_run_and_evaluate_returns_named_aligned_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared, "_load_splits", lambda **_kwargs: _splits())
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.single),
    )
    config = shared.parse_run_config(_config(dim=1, gaussian_params={"dim": 1}))

    result = ibu._run_and_evaluate(config, n_iterations=2)

    assert isinstance(result, ibu.IBUResult)
    assert result.variable_names == ("dim_0",)
    assert result.weights.shape == (1, 2)
    # Cast at IBU's boundary: the splits above are the float64 RAN trains on.
    assert result.weights.dtype == np.single
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status == "skipped"
    assert set(result.metrics) == {"detector_dim_0", "particle_dim_0"}
    for record in result.metrics.values():
        assert record["wasserstein_after"] == pytest.approx(
            record["wasserstein_before"]
        )


@pytest.mark.parametrize(
    ("n_iterations", "purity_threshold", "message"),
    [
        (0, ibu.DEFAULT_PURITY_THRESHOLD, "n_iterations"),
        (True, ibu.DEFAULT_PURITY_THRESHOLD, "n_iterations"),
        (1, np.double(np.nan), "purity_threshold"),
        (1, np.double(-0.1), "purity_threshold"),
        (1, np.single(1.1), "purity_threshold"),
    ],
)
def test_run_and_evaluate_validates_controls_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    n_iterations: int,
    purity_threshold: np.double,
    message: str,
) -> None:
    monkeypatch.setattr(
        shared,
        "_load_splits",
        lambda **_kwargs: pytest.fail("loaded data before validating controls"),
    )
    config = shared.parse_run_config(_config())

    with pytest.raises(ValueError, match=message):
        _ = ibu._run_and_evaluate(
            config,
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        )
