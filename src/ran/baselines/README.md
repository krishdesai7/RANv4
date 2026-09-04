# Baselines

This directory contains the comparison baselines for the RAN project.

Comparison baseline: **IBU**

```python
    from ran.baselines import evaluate_runs
```

## Shared

Module `._shared` holds the part of a baseline that is not the unfolding method: reading a run's config, rebuilding its populations, and scoring the resulting weights with the same metrics RAN is scored by.

A baseline attempts the same task RAN does — generate weights that reweight Generation, using only the relationship between Data and Simulation — so it needs the same run config, the same event populations, and the same metric record. Keeping those here means a comparison is a comparison of unfolding methods and nothing else. IBU is currently the only caller; the split is what makes adding a second one a matter of writing an unfolder.

### `_shared::parse_run_config`

Validate a run's config.json into a RunConfig.

#### Arguments

- `raw: dict[str, Any]` The raw config.json as a dictionary.

#### Returns

- A `RunConfig` object from the validated config.json.

### `_shared::_partitioned`

Checks the shape assumptions a baseline relies on, then partitions.

### function`prepare_populations(DatasetSplits, int) -> UnfoldingPopulations`

Returns an `UnfoldingPopulations`, which unpacks as `(fit, test)`. Both are `Populations`, and they are disjoint. `fit` is train+val and supplies the response (`fit.mc.z` and `fit.mc.x`, paired per event) and the measurement (`fit.data`). `test` is the held-out split alone, where the metrics are computed: detector level scores `test.data` against `test.mc.x`, particle level scores `test.truth` against `test.mc.z`. `test.truth` is the only place a baseline touches the answer key, and it appears only in scoring.

By construction, `fit` is `Split.TRAIN | Split.VAL`, not `Split.ALL`. A baseline fitted on every event and then
scored on the test split would be scored on data it had already used and would be handed information RAN is denied: `train` does read the test split now, to compute a test-level MMD diagnostic, but nothing weight-bearing depends on that read, so the test split still cannot influence the returned model or its selection (`tests/test_train.py::TestTrainingNeverSeesTheTestSplit`). The comparison is only a comparison if both sides see the same events.

Arrays arrive at the pipeline's pinned `EVENT_DTYPE` and are not cast here. IBU used to narrow to float32 at this boundary, to match the arithmetic its published results were produced with; now that the whole pipeline is float32 that cast is a no-op and is gone, along with the generics that existed to let the two precisions coexist. One thing still does widen: the two population-count checks and the mean-one postcondition accumulate in float64, because they compare against exact integer counts and float32 stops representing those past 2^24. Those are assertions about the data, not arithmetic on it.

#### Arguments

- `splits: DatasetSplits` The DatasetSplits object to partition.
- `expected_dim: int` The expected dimension of the dataset.

#### Returns

- An `UnfoldingPopulations` object containing the fit (train+val) and test populations.

### function `load_populations(RunConfig) -> UnfoldingPopulations`

Rebuild the run's dataset and split it into the baseline populations. It provides the run's dataset as populations, at the float64 RAN generated it in. Baselines that need another precision call `astype` at their own boundary.

#### Arguments

- `config: RunConfig` The RunConfig object to load the populations from.

#### Returns

- An `UnfoldingPopulations` object containing the fit (train+val) and test populations.

### `_shared::evaluate_dimension`

Score one dimension before and after reweighting `comparison`.

#### Arguments

- `reference: NDArray[np.floating]` The reference distribution.
- `comparison: NDArray[np.floating]` The comparison distribution.
- `weights: NDArray[np.floating]` The weights to apply to the comparison distribution.

#### Returns

- A `MetricRecord` containing the metrics.

## IBU

IBU (Iterative Bayesian Unfolding) baseline to compare with RAN. It is a simple unfolding method that uses a Bayesian approach to unfold the data.

It is implemented in the [**`ibu.py`**](ibu.py) file.
Usage:

```shell
ran baseline ibu --run-dir runs/2026-...
ran baseline ibu --run-dir runs # all runs
```

IBU performs 1D per-variable unfolding with purity-based automatic binning. It builds the response matrix from MC, unfolds data, and converts the result to per-event weights for evaluation with the same metrics as RAN.

## `class _BinnedReweighting`

A per-bin correction, learned from one population and applied to another.

IBU produces one multiplicative factor per bin of the particle-level axis. Which events it is then applied to is a separate choice: here the unfolding is fit on train+val and applied to the held-out test split, so the sample it scores is genuinely not the sample it learned from.

That is deliberately not what the unfolding literature usually does. Fitting the response and iterating the prior on every event, then quoting metrics on a subset of those same events, is conventional for both IBU and <span style="font-variant: small-caps;">OmniFold</span> — and it scores an estimator on data it has already seen. It also hands the baseline information RAN is denied: `ran.train` reads the test split only to compute a diagnostic that cannot influence the returned model, and never to fit or select. A comparison is only a comparison if both sides see the same events.

### `ibu::_assign_bins`

Assign every value to a saturated bin. Underflow enters the first bin; overflow and values equal to the upper edge enter the last bin, so every finite input value receives an assignment.

#### Arguments

- `values: NDArray[np.floating]` The values to assign to bins.
- `edges: NDArray[np.floating]` The bin edges.

#### Returns

- A `NDArray[np.intp]` containing the bin indices.

### `ibu::_bin_counts`

Count saturated assignments while preserving every assigned event. Assignments from `_assign_bins` place underflow in the first bin and both overflow and the upper edge in the last bin; their counts therefore retain one entry for every assigned event.

#### Arguments

- `indices: NDArray[np.intp]` The bin indices.
- `n_bins: int` The number of bins.

#### Returns

- A `NDArray[np.intp]` containing the bin counts.

### `ibu::_next_pure_edge`

Find the first candidate edge whose bin exceeds the purity threshold.

#### Arguments

- `gen_sorted: NDArray[np.floating]` The sorted gen values.
- `upper_sorted: NDArray[np.floating]` The sorted upper values.
- `lower_by_upper: NDArray[np.floating]` The lower values by upper values.
- `lo: np.floating` The lower edge.
- `gen_max: np.floating` The maximum gen value.
- `purity_threshold: float` The purity threshold.
- `n_candidates: int` The number of candidates.

#### Returns

- A `np.single | None` containing the first candidate edge whose bin exceeds the purity threshold.

### `ibu::_ibu`

Iterative Bayesian Unfolding.

#### Arguments

- `prior: NDArray[np.floating]` Initial truth estimate (MC gen histogram), shape (n_bins,).
- `data_hist: NDArray[np.floating]` Observed reco-level measured histogram, shape (n_bins,).
- `response: NDArray[np.floating]` R\[t,r\] = P(sim=r | gen=t), shape (n_bins, n_bins).
- `n_iterations: int` Number of unfolding iterations.
- `strict: bool = False` If True, raise an error if the observed data has zero support under the response and prior. If False \[default\], return zero weights for such events.

#### Returns

- A `EventArray` Unfolded truth histogram, shape (n_bins,).

### `ibu::unfold_variable`

Fit one variable's reweighting. Takes one column each of a `Populations`' `mc.z`, `mc.x` and `data`; those three are what a real measurement has, and `truth` is deliberately not among them. Returns a `VariableUnfolding`, which pairs the reweighting with a `VariableOutcome` recording whether the fit happened. Where purity binning yields fewer than two bins there is nothing to fit, so the reweighting is `None` and `weights_for` returns ones.

`mc_gen` and `mc_sim` are one column of a `Populations`' `mc.z` and `mc.x`; they are row-aligned and together give the response. `observed` is the same column of its `data`. No part of `truth` belongs here.

#### Arguments

- `variable_name: str` The variable being unfolded, for logging and the outcome record.
- `mc_gen: EventArray` One column of `mc.z`, the generated particle level.
- `mc_sim: EventArray` One column of `mc.x`, row-aligned with `mc_gen`; together they give the response.
- `observed: EventArray` One column of `data`, the measurement.
- `n_iterations: int` Number of unfolding iterations.
- `purity_threshold: float` The purity threshold for automatic binning.

#### Returns

- A `VariableUnfolding`, whose `weights_for(gen)` gives per-event weights for whichever sample is being scored.

### function `evaluate_single(Path, bool, int, float) -> dict[str, MetricRecord]`

Run IBU baseline on a single run. It fits on train+val, then scores the held-out test split with the result, so the sample scored is genuinely not the sample fitted.

#### Arguments

- `run_dir: Path` Path to a single run.
- `force: bool = False` Recompute even if metrics_ibu.json exists.
- `n_iterations: int = 10` Number of IBU iterations.
- `purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD` Purity threshold for automatic binning.

#### Returns

- A `dict[str, MetricRecord]` containing the metrics.

### `ibu::evaluate_runs`

Run IBU baseline on completed RAN runs.

#### Arguments

- `run_dir: Path` Path to a single run or directory of runs.
- `force: bool = False` Recompute even if metrics_ibu.json exists.
- `n_iterations: int = 10` Number of IBU iterations.
- `purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD` Purity threshold for automatic binning.
