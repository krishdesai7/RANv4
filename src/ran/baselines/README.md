# Baselines

This directory contains the baselines for the RAN project.

Comparison baselines: **IBU** and **OmniFold**.

Only the backend-agnostic helpers from `._shared` are re-exported. The two baseline modules are deliberately _not_ exported externally, because they need different Keras backends and there is one backend per interpreter. `.omnifold` hard-sets `KERAS_BACKEND=tensorflow` at import and must be the entry point of its own process, while `.ibu` reaches `ran.train`, which loads only on JAX. Re-exporting both would make each unimportable via the other and would leak the losing backend into every subprocess. Import the desired module directly

```python
    from ran.baselines.ibu import evaluate_runs
    from ran.baselines.omnifold import omnifold_unfold
```

## Shared

Module `._shared` contains the data handling shared by the IBU and OmniFold baselines.

Both methods attempt the same task: to generate weights to reweight Generation based on the relationship between Data and Simulation, and thus both baselines provide the same inputs and then score the result. So they need the same run config, the same event populations, and the same metric record. Only the unfolding method differs.

This module must stay free of `keras` at import time. `ran.baselines.omnifold` pins `KERAS_BACKEND=tensorflow` before importing keras, and it imports from here; pulling keras in transitively would fix the backend to jax first and break OmniFold. `ran.evaluate` is safe to import for the same reason, namely that it defers its own keras import into the two functions that need it.

### `_shared::parse_run_config`

Validate a run's config.json into a RunConfig.

#### Arguments

- `raw: dict[str, Any]` The raw config.json as a dictionary.

#### Returns

- A `RunConfig` object from the validated config.json.

### `_shared::_partitioned`

Checks the shape assumptions the baselines rely on, then partitions.

### `_shared::prepare_populations`

Returns an `UnfoldingPopulations`, which unpacks as `(full, test)`. Both are `Populations`. `full` spans every split and supplies the response (`full.mc.z` and `full.mc.x`, paired per event) and the measurement (`full.data`). `test` is the held-out split alone, where the metrics are computed: detector level scores `test.data` against `test.mc.x`, particle level scores `test.truth` against `test.mc.z`. `test.truth` is the only place a baseline touches the answer key, and it appears only in scoring.

Arrays maintain their dtype; a baseline that needs another dtype will cast at its own boundary with `Populations.astype`.

That is not a detail. RAN is float64 end to end, but both baselines are float32: OmniFold trains under TensorFlow, and IBU has to match the arithmetic its published results were produced with for the comparison to mean anything. So `load_populations` hands back the float64 the dataset was generated in, and each baseline narrows for itself — OmniFold in `_as2d`, IBU with `load_populations(config).astype(np.single)`. The IBU internals are generic over the floating type and carry whatever they are given; only the two population-count checks and the mean-one postcondition accumulate in float64, because those compare against exact integers and float32 stops representing those past 2^24.

#### Arguments

- `splits: DatasetSplits` The DatasetSplits object to partition.
- `expected_dim: int` The expected dimension of the dataset.

#### Returns

- An `UnfoldingPopulations` object containing the full and test populations.

### `_shared::load_populations`

Rebuild the run's dataset and split it into the baseline populations.

#### Arguments

- `config: RunConfig` The RunConfig object to load the populations from.

#### Returns

- An `UnfoldingPopulations` object containing the full and test populations.

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
uv run -m ran baseline ibu --run-dir runs/2026-...
uv run -m ran baseline ibu --run-dir runs # all runs
```

IBU performs 1D per-variable unfolding with purity-based automatic binning. It builds the response matrix from MC, unfolds data, and converts the result to per-event weights for evaluation with the same metrics as RAN and <span style="font-variant: small-caps;">OmniFold</span>.

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

- A `T | None` containing the first candidate edge whose bin exceeds the purity threshold.

### `ibu::_ibu`

Iterative Bayesian Unfolding.

#### Arguments

- `prior: NDArray[np.floating]` Initial truth estimate (MC gen histogram), shape (n_bins,).
- `data_hist: NDArray[np.floating]` Observed reco-level measured histogram, shape (n_bins,).
- `response: NDArray[np.floating]` R\[t,r\] = P(sim=r | gen=t), shape (n_bins, n_bins).
- `n_iterations: int` Number of unfolding iterations.
- `strict: bool = False` If True, raise an error if the observed data has zero support under the response and prior. If False \[default\], return zero weights for such events.

#### Returns

- A `NDArray[T]` Unfolded truth histogram, shape (n_bins,).

### `ibu::_BinnedReweighting`

What IBU actually produces: one multiplicative factor per bin of the particle-level axis, held with the edges that define those bins. `weights_for(gen)` looks up a factor for each particle-level value and renormalizes to mean one, so a reweighting can be applied to any number of events of the same variable.

Fitting and applying are separate because the two use different samples. The reweighting is fit on `full` — every split, for the largest response and measurement available — and applied to `test.mc.z`, so that the sample it scores is not the sample it learned from.

### `ibu::_unfold_variable`

Fit one variable's reweighting. Takes one column each of a `Populations`' `mc.z`, `mc.x` and `data`; those three are what a real measurement has, and `truth` is deliberately not among them. Returns a `_VariableUnfolding`, which pairs the reweighting with a `VariableOutcome` recording whether the fit happened. Where purity binning yields fewer than two bins there is nothing to fit, so the reweighting is `None` and `weights_for` returns ones.

#### Arguments

- `variable_name: str` The variable being unfolded, for logging and the outcome record.
- `mc_gen: NDArray[T]` One column of `mc.z`, the generated particle level.
- `mc_sim: NDArray[T]` One column of `mc.x`, row-aligned with `mc_gen`; together they give the response.
- `observed: NDArray[T]` One column of `data`, the measurement.
- `n_iterations: int` Number of unfolding iterations.
- `purity_threshold: float` The purity threshold for automatic binning.

#### Returns

- A `_VariableUnfolding[T]`, whose `weights_for(gen)` gives per-event weights for whichever sample is being scored.

## <span style="font-variant: small-caps;">OmniFold</span>

<span style="font-variant: small-caps;">OmniFold</span> baseline to compare with RAN. It is a deep learning-based unfolding method that uses a Bayesian approach to unfold the data.

It is implemented in the [**`omnifold.py`**](omnifold.py) file.
Usage:

```zsh
uv run -m ran baseline omnifold --run-dir runs/2026-...
uv run -m ran baseline omnifold --run-dir runs # all runs
```

RAN itself runs on the JAX backend, but the third-party `omnifold` package does not: its `weighted_binary_crossentropy` calls raw `tf.gather` on the label tensor, which raises `TracerArrayConversionError` the moment JAX traces it. So this module pins the backend back to TensorFlow.

A process gets one Keras backend, set at first `keras` import, so invoke the <span style="font-variant: small-caps;">OmniFold</span> baseline in its own process with `uv run -m ran baseline omnifold`; never import it from a module that has already touched JAX. The cubic sweep keeps the two sides in separate subcommands for exactly this reason.

### `ran.baselines.omnifold::omnifold_unfold`

Trains <span style="font-variant: small-caps;">OmniFold</span> on in-memory arrays and return mean-normalized gen weights.

Trains on (data reco = `x_data`, MC reco = `x_sim`, MC gen = `z_gen`), then reweights `z_target` (defaults to `z_gen`) through the gen-level model. Returns a 1D weight array, normalized so its mean is 1.
