# RAN Types

Records, constants and aliases shared across the package.

These live apart from the code that uses them because a process gets a single Keras backend, fixed at the first `keras` import, so `ran.cli` and `ran.baselines._shared` have to stay importable without committing to one. They cannot reach into `ran.train` (JAX) or `ran.baselines.omnifold` (TensorFlow) for a shared declaration, so the
declaration must live here instead. Nothing in this package imports keras or jax
at runtime.

Types owned by exactly one module stay with that module. E.g., `TrainResult` and `TrainState` are in `ran.train`.

## module: `configs`

Validated views of a run's Gaussian parameter set and `config.json` that together configure a run.

### `class RunConfig`

A validated view of a run's `config.json`. `source` is the raw dict, kept because `_load_splits` reconstructs the dataset from it and must see exactly what the run recorded.

#### Fields

- `source: dict[str, Any]`: The raw config.json.
- `dataset: DatasetName`: The dataset to use.
- `dim: int`: The dimension of the Gaussian parameter set.
- `n_samples: int`: The number of samples to use.
- `batch_size: int`: The batch size to use.
- `data_seed: int`: The seed for the dataset.
- `variable_names: tuple[str, ...]`: The names of the jet variables to use.

## module: `constants`

Fixed values: the Zenodo jet dataset, its cache layout, plot metadata, the default purity threshold, and the stand-in for an absent particle level.

### Constant: `LOG_RHO_FLOOR: Final[float]`

Soft drop grooms some jets down to a single prong, leaving $m_{sd} = 0$ and $\ln(\rho) = \ln\left(\frac{m_{sd}^2}{p_T^2}\right) = -\infty$. Those jets take this value instead, which is both the bottom of the plotted range below and the reason it can be: no jet with a groomed mass reaches it.

Furthermore, unlike $-\infty$ (or the $10^{-100}$ the upstream <span style="font-variant: small-caps;">OmniFold</span> observable adds inside the log, which floors at $-230$) it does not swamp the mean and variance that the features are standardized by. Degenerate jets land in the underflow bin, where they belong.

### Constant: `TRUTH_SENTINEL: Final[np.double]`

Stands in for a particle-level value that does not exist, in the one place that happens: a real measurement, which has data but no answer key.

It has to be finite. `normalize_weights` annihilates the nature rows of the generator's output by multiplying by $(1 - y) = 0$, and under IEEE 754 that annihilates any finite number but not a NaN. `0 * np.nan` is `np.nan`, which then spreads through the class-normalizing sum to every weight in the batch, and through `jax.grad` to every gradient. Sanitizing the masked output would not help either, since the `np.nan` is already in `z` when `g` forward-passes it. The fix has to sit at the input, and be an ordinary number.

$-2^{15}$ is that number: absurd on sight for any standardized observable, exact in every IEEE binary format down to float16, and far enough inside float16's range (65504) that it survives a narrowing cast instead of becoming `±np.infty`, which would put `0 * np.infty = np.nan` right back.

## module: `enums`

CLI choice enums. `ran.cli` imports these while it is still backend-free, which is why they live in this module rather than beside the code they select for.

## module: `events`

The event data model, in its two representations.

`Populations` is the view that represents the physical sources of events, and `ZXY` that represents the events as they exist in a ML pipeline. Conversion between the two is not strictly lossless. Converting from `ZXY` to populations discards ordering information. Converting from `Populations` to `ZXY` is, however, lossless.

The various event dataclasses take `eq=False` because they hold arrays: a generated `__eq__` compares fields with `==` and a generated `__hash__` hashes them. Both operations raise when called on arrays. They can only be compared for identity.

### `class Events`

A corresponding pair of (particle-level `z`, detector-level `x`) for one set of events. The arrays are row-aligned: row `i` of each is the same event seen at the two levels.

#### Fields

- `z: NDArray[T]`: The particle-level features.
- `x: NDArray[T]`: The detector-level features.

#### Methods

- `concatenate(parts: Sequence[Events[T]]) -> Events[T]`: Concatenate a sequence of events.
- `astype[U: np.floating](self, dtype: type[U]) -> Events[U]`: Cast the event arrays to a different dtype.

### `class Populations`

The physics view of a labelled sample.

`mc` is the simulation, its particle level generation (`mc.z`) paired per event with the corresponding detector level simulation (`mc.x`); that pairing is used to build a response matrix.

`data` is the natural measurement. `truth` is the particle-level answer key. It exists only because every dataset here is a closure test. A real measurement has no such array, and no network may ever see it. Keeping it out of `mc` means a function handed the MCMC cannot reach it. Construct through `create` to set `truth` to `ran.rantypes.constants.TRUTH_SENTINEL`; the field itself is always present.

#### Fields

- `mc: Events[T]`: The simulation.
- `data: NDArray[T]`: The natural measurement.
- `truth: NDArray[T]`: The particle-level answer key.

#### Properties

- `has_truth: bool`: `True` if the sample has a particle-level answer key. `False` if it is populated with `ran.rantypes.constants.TRUTH_SENTINEL`.

Any metric computed against a sentinel `truth` is meaningless but finite, so unfolding code that scores against the particle level has to ask rather than wait to be told.

#### Methods

##### `create(mc: Events[T], data: NDArray[T], truth: NDArray[T] | None = None) -> Populations[T]`

Build a sample, filling `truth` with `ran.rantypes.constants.TRUTH_SENTINEL` if there is none.

A real measurement has no answer key. Filling the field rather than dropping it keeps one type for both cases, and keeps the sample trainable: the nature rows of `z` are `truth`, so they reach the generator, and only a finite value there lets `normalize_weights` annihilate them as intended. See `ran.rantypes.constants.TRUTH_SENTINEL` for why not NaN.

`truth` is particle level, so it takes its columns from `mc.z` and its rows from `data`.

**Arguments:**

- `mc: Events[T]`: The simulation.
- `data: NDArray[T]`: The natural measurement.
- `truth: NDArray[T] | None = None`: The particle-level answer key. If `None`, `TRUTH_SENTINEL` is used.

**Returns:**

- `Populations[T]`: The sample.

##### `astype[U: np.floating](self, dtype: type[U]) -> Populations[U]`

The same sample at another precision. RAN operates by default at float64 precision end to end, but allows casting to other dtypes if required. Any method that requires a precision distinct from float64 must cast at its own boundary rather than enforcing a shared precision for the entire pipeline. `ran.rantypes.constants.TRUTH_SENTINEL` is exact in every IEEE binary format, so `has_truth` answers the same question on either side of this call.

**Arguments:**

- `dtype: type[U]`: The dtype to cast the sample to.

**Returns:**

- `Populations[U]`: The sample at the new precision.

##### `require_truth() -> NDArray[T]`

Returns `truth` if available or raises a `ValueError` if there is none. Scoring against the sentinel yields a finite, meaningless number instead of an obvious failure, so the particle-level comparisons ask for the answer key through here rather than reading the field.

**Returns:**

- `NDArray[T]`: The particle-level answer key.

##### `interleave() -> ZXY[T]`

Stack into the labelled transport form, nature rows first. The resulting row order is an artifact of stacking rather than anything meaningful, so callers shuffle before splitting.

**Returns:**

- `ZXY[T]`: The stacked events.

### `class ZXY`

Events labelled by provenance: y = 1 for nature, y = 0 for MC. The form in which they get shuffled, split, batched and trained on. `partition` converts to the physics form, `Populations.interleave` back.

#### Fields

- `events: Events[T]`: The events.
- `y: NDArray[np.ubyte]`: The labels.

#### Properties

- `z: NDArray[T]`: The particle-level features.
- `x: NDArray[T]`: The detector-level features.

### `DatasetSplits::select(which: Split = Split.ALL) -> ZXY[T]`

Concatenate the requested splits into one labelled sample. The split a row came from is not recorded on the result: it is a property of the query, not of the events, and nothing downstream of this call can act on it.

**Arguments:**

- `which: Split = Split.ALL`: The splits to concatenate.

**Returns:**

- `ZXY[T]`: The concatenated events.

## module: `results`

### `class UnfoldingPopulations`

The input to an unfolding run: a sample to unfold with, and a sample to score on.

`full` spans every split and supplies the response (`full.mc`) and the measurement (`full.data`). `test` is the held-out split alone, which is where the metrics are computed.

#### Fields

- `full: Populations[T]`: The sample to unfold with.
- `test: Populations[T]`: The sample to score on.
