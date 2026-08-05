# IBU Evaluation Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic IBU evaluation function with a typed, testable pipeline that uses one saturating bin policy, validates configuration and arrays early, produces safe normalized weights, and reports explicit per-variable outcomes.

**Architecture:** `src/ran/baselines/ibu.py` remains the sole implementation module. Immutable dataclasses define configuration, prepared data, per-variable outcomes, and the final result; pure helpers handle bin assignment, counts, numerical validation, per-variable unfolding, and scalar metric evaluation; `_run_and_evaluate` only coordinates them. Existing metric JSON and weight archive formats remain unchanged.

**Tech Stack:** Python 3.13, NumPy 2.4, pytest, Ruff, Pyrefly, Complexipy.

## Global Constraints

- Keep the implementation internal to `src/ran/baselines/ibu.py`; do not refactor OmniFold or shared evaluation modules.
- Preserve commit `1aea4cf`, including the optimized purity binning and baseline
  README, plus the user's current `NDArray` annotation correction in
  `src/ran/baselines/ibu.py`.
- Do not stage or modify the user's unrelated current changes in
  `src/ran/baselines/README.md` or `src/ran/baselines/omnifold.py`.
- Use saturating first and last bins for response MC, prior counts, observed-data counts, and test-MC lookup.
- Preserve the flat `metrics_ibu.json` schema and `weights_<index>` keys in `ibu_weights.npz`.
- Keep metric definitions and purity-edge search semantics unchanged.
- Represent final weights as one `np.double` array with shape `(n_variables, n_test_mc)`.
- Give every variable detector and particle metric records, including skipped variables with identity weights.
- Distinguish insufficient-bin skips from numerical invariant failures.
- Reject nonfinite arrays, invalid zero-prior mass, negative weights, and non-normalizable weights with descriptive `ValueError`s.
- Keep every new or materially changed function at cyclomatic complexity 10 or below.
- Implement each behavior test-first and commit each task independently.

---

### Task 1: Typed Configuration and Result Contracts

**Files:**
- Modify: `src/ran/baselines/ibu.py:1-35,191-195`
- Create: `tests/test_ibu.py`

**Interfaces:**
- Produces: `_parse_config(raw: object) -> IBUConfig`
- Produces: `MetricRecord`, `VariableOutcome`, and `IBUResult`
- Produces: `IBUConfig.variable_names: tuple[str, ...]`
- Consumes: the existing saved-run JSON fields `dataset`, `dim`, `n_samples`, `batch_size`, `data_seed`, and `variables`

- [ ] **Step 1: Add failing configuration-contract tests**

Create `tests/test_ibu.py` with the following foundation and tests:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ran.baselines import ibu

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


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
    parsed = ibu._parse_config(_config())

    assert parsed.dataset == "gaussian"
    assert parsed.variable_names == ("dim_0", "dim_1")
    assert parsed.dim == 2
    assert parsed.data_seed == 7


def test_parse_config_requires_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        ibu._parse_config(["not", "a", "mapping"])


def test_parse_config_validates_jet_variable_count() -> None:
    with pytest.raises(ValueError, match="variables.*dim"):
        ibu._parse_config(
            _config(dataset="jets", variables=["mass"], dim=2)
        )


@pytest.mark.parametrize("key", ["dim", "n_samples", "batch_size"])
def test_parse_config_requires_positive_integer_fields(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        ibu._parse_config(_config(**{key: 0}))


def test_parse_config_reports_missing_required_field() -> None:
    config = _config()
    del config["n_samples"]

    with pytest.raises(ValueError, match="n_samples"):
        ibu._parse_config(config)


def test_parse_config_rejects_non_integer_data_seed() -> None:
    with pytest.raises(ValueError, match="data_seed"):
        ibu._parse_config(_config(data_seed="seven"))


def test_parse_config_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        ibu._parse_config(_config(dataset="other"))


def test_parse_config_rejects_non_string_jet_variable() -> None:
    with pytest.raises(ValueError, match="variables.*strings"):
        ibu._parse_config(
            _config(dataset="jets", variables=["mass", 4], dim=2)
        )
```

- [ ] **Step 2: Run the focused tests and verify the missing-interface failure**

Run: `uv run pytest -q tests/test_ibu.py`

Expected: FAIL because `ran.baselines.ibu` has no `_parse_config` or `IBUConfig`.

- [ ] **Step 3: Add typed dataclasses and configuration parsing**

In `src/ran/baselines/ibu.py`, import `dataclass` and `TypedDict` at runtime and add these contracts near the module constants:

```python
class MetricRecord(TypedDict):
    wasserstein_before: float
    wasserstein_after: float
    wasserstein_improvement_pct: float
    jensenshannon_before: float
    jensenshannon_after: float
    jensenshannon_improvement_pct: float
    triangular_before: float
    triangular_after: float
    triangular_improvement_pct: float


@dataclass(frozen=True)
class IBUConfig:
    source: dict[str, Any]
    dataset: Literal["gaussian", "jets"]
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]


@dataclass(frozen=True)
class VariableOutcome:
    variable_name: str
    status: Literal["completed", "skipped"]
    n_bins: int
    skip_reason: str | None = None


@dataclass(frozen=True)
class IBUResult:
    metrics: dict[str, MetricRecord]
    variable_names: tuple[str, ...]
    weights: NDArray[np.double]
    outcomes: tuple[VariableOutcome, ...]
```

Add `_positive_int` and `_parse_config`. Reject a non-dictionary root with
`"IBU config must be a JSON object"`. Use `type(value) is int` so JSON booleans
are not accepted as integers. Preserve the legacy default dataset of
`"gaussian"` and data seed of `42`. Accept a list or tuple of jet variable
names, require every name to be a nonempty string, and require
`len(variable_names) == dim`. Store a shallow `dict(raw)` copy in
`IBUConfig.source`.

Use these exact error-message fragments so the tests and CLI diagnostics stay stable:

```python
f"{key} must be a positive integer"
"data_seed must be an integer"
f"Unknown dataset: {dataset!r}"
"variables must be a sequence of nonempty strings"
f"variables has length {len(variable_names)}, expected dim={dim}"
```

- [ ] **Step 4: Run and format the focused implementation**

Run: `uv run pytest -q tests/test_ibu.py`

Expected: all Task 1 tests PASS.

Run: `uv run ruff format src/ran/baselines/ibu.py tests/test_ibu.py`

Run: `uv run ruff check src/ran/baselines/ibu.py tests/test_ibu.py`

Expected: both Ruff commands exit successfully.

- [ ] **Step 5: Commit the typed boundary**

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py
git commit -m "refactor: type IBU configuration and results"
```

---

### Task 2: Prepared Inputs and Consistent Saturating Bins

**Files:**
- Modify: `src/ran/baselines/ibu.py:191-224`
- Modify: `tests/test_ibu.py`

**Interfaces:**
- Consumes: `DatasetSplits`, `IBUConfig.dim`, existing `_build_response`
- Produces: `_IBUData`
- Produces: `_prepare_data(splits: DatasetSplits, expected_dim: int) -> _IBUData`
- Produces: `_assign_bins(values: NDArray[np.double], edges: NDArray[np.double]) -> NDArray[np.intp]`
- Produces: `_bin_counts(indices: NDArray[np.intp], n_bins: int) -> NDArray[np.double]`

- [ ] **Step 1: Add failing population and array-validation tests**

Extend `tests/test_ibu.py` with imports for `ArrayDataset` and `DatasetSplits`, plus this fixture helper:

```python
from ran.data import ArrayDataset, DatasetSplits


def _split(
    z: list[list[float]], x: list[list[float]], y: list[int]
) -> ArrayDataset:
    return ArrayDataset(
        np.asarray(z, dtype=np.double),
        np.asarray(x, dtype=np.double),
        np.asarray(y, dtype=np.ubyte),
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
```

Add these tests:

```python
def test_assign_bins_saturates_underflow_and_overflow() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.double)
    values = np.array([-3.0, 0.0, 0.4, 1.0, 2.0, 8.0], dtype=np.double)

    indices = ibu._assign_bins(values, edges)

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(ibu._bin_counts(indices, 2), [3.0, 3.0])


def test_saturating_counts_conserve_observed_population() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.double)
    observed = np.array([-2.0, 0.5, 4.0], dtype=np.double)

    counts = ibu._bin_counts(ibu._assign_bins(observed, edges), 2)

    assert counts.sum() == observed.size
    np.testing.assert_array_equal(counts, [2.0, 1.0])


def test_prepare_data_names_response_and_test_populations() -> None:
    data = ibu._prepare_data(_splits(), expected_dim=1)

    assert data.response_gen.shape == (5, 1)
    assert data.response_sim.shape == (5, 1)
    assert data.observed_reco.shape == (5, 1)
    assert data.test_mc_gen.shape == (2, 1)
    assert data.test_data_reco.shape == (2, 1)


def test_prepare_data_rejects_configured_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="expected dim=2"):
        ibu._prepare_data(_splits(), expected_dim=2)


def test_prepare_data_rejects_nonfinite_values() -> None:
    splits = _splits()
    splits.train.x[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        ibu._prepare_data(splits, expected_dim=1)


def test_prepare_data_rejects_empty_test_data_population() -> None:
    splits = _splits()
    without_test_data = DatasetSplits(
        train=splits.train,
        val=splits.val,
        test=_split([[0.8], [1.8]], [[0.9], [2.5]], [0, 0]),
    )

    with pytest.raises(ValueError, match="test data"):
        ibu._prepare_data(without_test_data, expected_dim=1)
```

- [ ] **Step 2: Run the new tests and verify missing-helper failures**

Run: `uv run pytest -q tests/test_ibu.py -k 'assign_bins or saturating_counts or prepare_data'`

Expected: FAIL because `_assign_bins`, `_bin_counts`, and `_prepare_data` do not exist.

- [ ] **Step 3: Implement named input preparation and saturating counts**

Add the immutable `_IBUData` dataclass with fields:

```python
response_gen: NDArray[np.double]
response_sim: NDArray[np.double]
observed_reco: NDArray[np.double]
test_data_gen: NDArray[np.double]
test_data_reco: NDArray[np.double]
test_mc_gen: NDArray[np.double]
test_mc_reco: NDArray[np.double]
```

Implement `_prepare_data` by calling `as_arrays()` once for each split,
validating each `z`/`x` pair is finite, two-dimensional, identically shaped,
has `expected_dim` columns, and shares its row count with one-dimensional `y`.
Require labels to be only zero or one. Concatenate the three validated splits,
derive the named MC/data arrays, and reject an empty response-MC, observed-data,
test-MC, or test-data population.

Implement bin helpers with these semantics:

```python
def _assign_bins(values, edges):
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("bin values must be a finite one-dimensional array")
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bin edges must be a strictly increasing 1D array")
    n_bins = edges.size - 1
    return (np.clip(np.digitize(values, edges), 1, n_bins) - 1).astype(
        np.intp, copy=False
    )


def _bin_counts(indices, n_bins):
    if n_bins < 1 or indices.ndim != 1:
        raise ValueError("bin indices must be one-dimensional with n_bins >= 1")
    if np.any((indices < 0) | (indices >= n_bins)):
        raise ValueError("bin index outside configured range")
    return np.bincount(indices, minlength=n_bins).astype(np.double)
```

Do not yet rewrite `_run_and_evaluate`; Task 4 performs that integration after
the numerical helpers have direct test coverage.

- [ ] **Step 4: Verify Task 2 behavior and static checks**

Run: `uv run pytest -q tests/test_ibu.py`

Run: `uv run ruff format src/ran/baselines/ibu.py tests/test_ibu.py`

Run: `uv run ruff check src/ran/baselines/ibu.py tests/test_ibu.py`

Expected: all commands succeed.

- [ ] **Step 5: Commit input and bin semantics**

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py
git commit -m "fix: unify IBU bin population semantics"
```

---

### Task 3: Safe Per-Variable Unfolding and Scalar Metrics

**Files:**
- Modify: `src/ran/baselines/ibu.py:164-188,236-325`
- Modify: `tests/test_ibu.py`

**Interfaces:**
- Consumes: `_purity_bins`, `_assign_bins`, `_bin_counts`, `_build_response`, `_ibu`, `MetricRecord`, and `VariableOutcome`
- Produces: `_VariableUnfolding`
- Produces: `_unfolded_to_bin_weights(unfolded, prior) -> NDArray[np.double]`
- Produces: `_normalize_weights(weights) -> NDArray[np.double]`
- Produces: `_unfold_variable(...) -> _VariableUnfolding`
- Produces: `_evaluate_dimension(reference, comparison, weights) -> MetricRecord`

- [ ] **Step 1: Add failing numerical-invariant tests**

Add these tests to `tests/test_ibu.py`:

```python
def test_unfolded_to_bin_weights_uses_explicit_zero_prior_semantics() -> None:
    weights = ibu._unfolded_to_bin_weights(
        np.array([4.0, 0.0], dtype=np.double),
        np.array([2.0, 0.0], dtype=np.double),
    )

    np.testing.assert_array_equal(weights, [2.0, 0.0])


def test_unfolded_to_bin_weights_rejects_mass_without_prior() -> None:
    with pytest.raises(ValueError, match="zero-prior"):
        ibu._unfolded_to_bin_weights(
            np.array([4.0, 1.0], dtype=np.double),
            np.array([2.0, 0.0], dtype=np.double),
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
    weights: NDArray[np.double], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ibu._normalize_weights(weights)


def test_normalize_weights_returns_mean_one() -> None:
    normalized = ibu._normalize_weights(
        np.array([1.0, 2.0, 3.0], dtype=np.double)
    )

    assert normalized.mean() == pytest.approx(1.0)
    assert np.all(normalized >= 0)
```

- [ ] **Step 2: Run the invariant tests and verify missing-helper failures**

Run: `uv run pytest -q tests/test_ibu.py -k 'bin_weights or normalize_weights'`

Expected: FAIL because the safe ratio and normalization helpers do not exist.

- [ ] **Step 3: Implement explicit ratio and normalization helpers**

`_unfolded_to_bin_weights` must require matching nonempty one-dimensional
shapes, finite nonnegative inputs, and no unfolded value greater than `EPS` in
a zero-prior bin. Initialize `weights = np.zeros_like(unfolded)` and call
`np.divide(unfolded, prior, out=weights, where=prior > 0)`.

`_normalize_weights` must require a nonempty one-dimensional finite,
nonnegative vector. Compute the mean once and reject it unless it is finite and
strictly positive. Divide, then verify the result is finite, nonnegative, and
`np.isclose(normalized.mean(), 1.0)`. Return an `np.double` array.

- [ ] **Step 4: Add failing skip, success, and scalar-metric tests**

```python
def test_unfold_variable_reports_insufficient_bins_as_skip(monkeypatch) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.double),
    )

    result = ibu._unfold_variable(
        variable_name="dim_0",
        response_gen=np.array([0.2, 0.8], dtype=np.double),
        response_sim=np.array([0.3, 0.7], dtype=np.double),
        observed_reco=np.array([-1.0, 2.0], dtype=np.double),
        test_mc_gen=np.array([0.4, 0.6], dtype=np.double),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.outcome.status == "skipped"
    assert result.outcome.skip_reason == "fewer than two purity bins"
    np.testing.assert_array_equal(result.weights, [1.0, 1.0])


def test_unfold_variable_returns_safe_mean_one_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.double),
    )

    result = ibu._unfold_variable(
        variable_name="dim_0",
        response_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.double),
        response_sim=np.array([-1.0, 0.7, 1.3, 3.0], dtype=np.double),
        observed_reco=np.array([-2.0, 0.4, 1.4, 4.0], dtype=np.double),
        test_mc_gen=np.array([0.2, 1.8], dtype=np.double),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.outcome.status == "completed"
    assert result.outcome.n_bins == 2
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0)
    assert result.weights.mean() == pytest.approx(1.0)


def test_evaluate_dimension_accepts_one_dimensional_arrays() -> None:
    record = ibu._evaluate_dimension(
        reference=np.array([0.0, 1.0, 2.0], dtype=np.double),
        comparison=np.array([0.0, 1.5, 3.0], dtype=np.double),
        weights=np.ones(3, dtype=np.double),
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
    assert all(np.isfinite(value) for value in record.values())
```

- [ ] **Step 5: Implement the per-variable and metric helpers**

Add `_VariableUnfolding` as an immutable dataclass with `weights` and `outcome`
fields. Implement `_unfold_variable` with the exact signature exercised above:

1. Call `_purity_bins` on response generator and simulation values.
2. If fewer than two bins result, log the warning and return identity weights
   plus `VariableOutcome(variable_name, "skipped", n_bins,
   "fewer than two purity bins")`.
3. Assign response generator, response simulation, observed reconstructed data,
   and test generator values with `_assign_bins`.
4. Build the response with `_build_response`; build prior and observed counts
   with `_bin_counts`.
5. Assert prior and observed sums equal their source event counts.
6. Run `_ibu`, convert with `_unfolded_to_bin_weights`, select test weights,
   normalize with `_normalize_weights`, and return a completed outcome.

Implement `_evaluate_dimension` by calling the existing three per-dimension
metric functions directly with one-dimensional arrays, taking element zero from
each returned list, and constructing all nine `MetricRecord` fields with the
existing `_improvement` helper.

- [ ] **Step 6: Run all focused tests and quality checks**

Run: `uv run pytest -q tests/test_ibu.py`

Run: `uv run ruff format src/ran/baselines/ibu.py tests/test_ibu.py`

Run: `uv run ruff check src/ran/baselines/ibu.py tests/test_ibu.py`

Expected: all commands succeed.

- [ ] **Step 7: Commit safe per-variable unfolding**

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py
git commit -m "refactor: isolate safe IBU variable unfolding"
```

---

### Task 4: Short Orchestrator and Artifact Integration

**Files:**
- Modify: `src/ran/baselines/ibu.py:191-379`
- Modify: `tests/test_ibu.py`
- Modify: `tests/test_completion_logging.py:119-150`

**Interfaces:**
- Consumes: `IBUConfig`, `_IBUData`, `_unfold_variable`, `_evaluate_dimension`, and `IBUResult`
- Produces: `_run_and_evaluate(config: IBUConfig, n_iterations: int = 10, purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD) -> IBUResult`
- Preserves: `evaluate_single(...) -> dict[str, MetricRecord]`
- Preserves: `metrics_ibu.json` flat metric keys and `ibu_weights.npz` row keys

- [ ] **Step 1: Add a failing orchestrator result test**

Add this test to `tests/test_ibu.py`:

```python
def test_run_and_evaluate_returns_named_aligned_result(monkeypatch) -> None:
    monkeypatch.setattr(ibu, "_load_splits", lambda _config: _splits())
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.double),
    )
    config = ibu._parse_config(
        _config(dim=1, gaussian_params={"dim": 1})
    )

    result = ibu._run_and_evaluate(config, n_iterations=2)

    assert isinstance(result, ibu.IBUResult)
    assert result.variable_names == ("dim_0",)
    assert result.weights.shape == (1, 2)
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status == "skipped"
    assert set(result.metrics) == {"detector_dim_0", "particle_dim_0"}
    for record in result.metrics.values():
        assert record["wasserstein_after"] == pytest.approx(
            record["wasserstein_before"]
        )
```

- [ ] **Step 2: Run the orchestration test and verify the return-shape failure**

Run: `uv run pytest -q tests/test_ibu.py::test_run_and_evaluate_returns_named_aligned_result`

Expected: FAIL because `_run_and_evaluate` still expects a raw dictionary and
returns a tuple of a metric dictionary, names, and a weight list.

- [ ] **Step 3: Rewrite `_run_and_evaluate` as an orchestrator**

Replace its body with this data flow:

```python
splits = _load_splits(config.source)
data = _prepare_data(splits, config.dim)
weights = np.empty((config.dim, data.test_mc_gen.shape[0]), dtype=np.double)
metrics: dict[str, MetricRecord] = {}
outcomes: list[VariableOutcome] = []

for dimension, variable_name in enumerate(config.variable_names):
    unfolding = _unfold_variable(
        variable_name=variable_name,
        response_gen=data.response_gen[:, dimension],
        response_sim=data.response_sim[:, dimension],
        observed_reco=data.observed_reco[:, dimension],
        test_mc_gen=data.test_mc_gen[:, dimension],
        n_iterations=n_iterations,
        purity_threshold=purity_threshold,
    )
    weights[dimension] = unfolding.weights
    outcomes.append(unfolding.outcome)
    metrics[f"detector_{variable_name}"] = _evaluate_dimension(
        data.test_data_reco[:, dimension],
        data.test_mc_reco[:, dimension],
        unfolding.weights,
    )
    metrics[f"particle_{variable_name}"] = _evaluate_dimension(
        data.test_data_gen[:, dimension],
        data.test_mc_gen[:, dimension],
        unfolding.weights,
    )

return IBUResult(
    metrics=metrics,
    variable_names=config.variable_names,
    weights=weights,
    outcomes=tuple(outcomes),
)
```

Validate `n_iterations` is a positive integer and `purity_threshold` is finite
and between zero and one before loading data. Keep this function limited to
validation and orchestration; do not move numerical implementation back into
it.

- [ ] **Step 4: Update `evaluate_single` and its completion test double**

In `evaluate_single`, parse the loaded JSON once:

```python
raw_config: object = json.loads((run_dir / "config.json").read_text())
config = _parse_config(raw_config)
result = _run_and_evaluate(
    config,
    n_iterations=n_iterations,
    purity_threshold=purity_threshold,
)
```

Write `result.metrics` to JSON, pass `list(result.variable_names)` to
`render_metrics`, return `result.metrics`, and preserve the existing NPZ key
shape with:

```python
**{f"weights_{i}": weights for i, weights in enumerate(result.weights)}
```

Update `tests/test_completion_logging.py` so `fake_run_and_evaluate` accepts an
`IBUConfig` and returns:

```python
ibu.IBUResult(
    metrics={},
    variable_names=("dim_0",),
    weights=np.ones((1, 2), dtype=np.double),
    outcomes=(
        ibu.VariableOutcome(
            variable_name="dim_0",
            status="completed",
            n_bins=2,
        ),
    ),
)
```

The test's `{}` config fixture is no longer valid at the parsing boundary.
Write a minimal valid Gaussian config containing `dim`, `n_samples`, and
`batch_size` before invoking `evaluate_single`.

- [ ] **Step 5: Run focused orchestration and artifact tests**

Run: `uv run pytest -q tests/test_ibu.py tests/test_completion_logging.py`

Expected: all focused tests PASS, including both metric and weight artifact
completion assertions.

- [ ] **Step 6: Check complexity and typing of the completed module**

Run: `uv run ruff format src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py`

Run: `uv run ruff check src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py`

Run: `uv run pyrefly check src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py`

Run: `uv run complexipy src/ran/baselines/ibu.py`

Expected: all commands succeed and no function exceeds complexity 10.

- [ ] **Step 7: Commit orchestration integration**

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py
git commit -m "refactor: compose typed IBU evaluation pipeline"
```

---

### Task 5: Full Regression Verification

**Files:**
- Modify only if a formatter makes mechanical changes: `src/ran/baselines/ibu.py`, `tests/test_ibu.py`, `tests/test_completion_logging.py`

**Interfaces:**
- Consumes: the completed IBU pipeline and all repository validation tools
- Produces: evidence that the refactor is regression-free and preserves caller artifacts

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all repository tests pass, with only the repository's existing skip if
it remains applicable.

- [ ] **Step 2: Run repository-wide formatting and lint checks**

Run: `uv run ruff format --check .`

Run: `uv run ruff check .`

Expected: both commands succeed without changing unrelated user files.

- [ ] **Step 3: Run repository-wide type and complexity checks**

Run: `uv run pyrefly check .`

Run: `uv run complexipy .`

Expected: both commands succeed. If an unrelated pre-existing failure is
present, record the exact command and diagnostic separately rather than editing
out-of-scope files.

- [ ] **Step 4: Inspect the final diff and artifact compatibility**

Run: `git diff 1aea4cf..HEAD -- src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py`

Confirm from the diff that:

- the pre-existing one-line `NDArray` annotation correction remains intact;
- `_run_and_evaluate` contains orchestration but no histogram construction or
  metric-field assembly;
- `np.histogram` is no longer used by the IBU computation;
- no `unfolded / (prior + EPS)` or unchecked `w / w.mean()` remains;
- `IBUResult.weights` is a two-dimensional array;
- skipped variables receive both metric records and an explicit outcome; and
- NPZ keys remain `weights_<index>`.

- [ ] **Step 5: Commit any verification-only formatting changes**

If Step 2 produced tracked formatting changes within the three in-scope files,
run:

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py tests/test_completion_logging.py
git commit -m "style: finalize IBU evaluation refactor"
```

If Step 2 produced no changes, do not create an empty commit.
