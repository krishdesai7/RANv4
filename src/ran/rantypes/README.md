# RAN Types

Records, constants and aliases shared across the package.

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

### Constants: `CACHE_ENV_VAR`, `CACHE_DIR`, `COMPILE_CACHE_DIR`

Everything RAN can regenerate shares one root: generated Gaussian datasets, the per-variable jet caches pulled from Zenodo, and the XLA compilation cache under `jax/`. `CACHE_DIR` is `.cache` unless `RAN_CACHE_DIR` (the value of `CACHE_ENV_VAR`) says otherwise, which is what relocates the tree to `$SCRATCH` on a cluster where `$HOME` is quota'd and shared.

It is deliberately not derived from `XDG_CACHE_HOME`. That variable is already set, or defaults to `~/.cache`, on most Linux systems — deriving from it would silently move every existing checkout's cache and orphan the jet data already on disk.

`~` is expanded, and an empty value falls back to the default rather than being taken as the current directory: a SLURM `--export` forwarding an unset variable delivers `""`, not absence. The value is read once, at import, because the `cache_dir=` defaults throughout `ran.data` bind to `CACHE_DIR` at import either way.

### Constant: `LOG_RHO_FLOOR: Final[float]`

Soft drop grooms some jets down to a single prong, leaving $m_{sd} = 0$ and $\ln(\rho) = \ln\left(\frac{m_{sd}^2}{p_T^2}\right) = -\infty$. Those jets take this value instead, which is both the bottom of the plotted range below and the reason it can be: no jet with a groomed mass reaches it.

Furthermore, unlike $-\infty$ (or the $10^{-100}$ the upstream <span style="font-variant: small-caps;">OmniFold</span> observable adds inside the log, which floors at $-230$) it does not swamp the mean and variance that the features are standardized by. Degenerate jets land in the underflow bin, where they belong.

### Constant: `TRUTH_SENTINEL: Final[np.double]`

Stands in for a particle-level value that does not exist, in the one place that happens: a real measurement, which has data but no answer key.

It has to be finite. `normalize_weights` annihilates the nature rows of the generator's output by multiplying by $(1 - y) = 0$, and under IEEE 754 that annihilates any finite number but not a NaN. `0 * np.nan` is `np.nan`, which then spreads through the class-normalizing sum to every weight in the batch, and through `jax.grad` to every gradient. Sanitizing the masked output would not help either, since the `np.nan` is already in `z` when `g` forward-passes it. The fix has to sit at the input, and be an ordinary number.

$-2^{15}$ is that number: absurd on sight for any standardized observable, exact in every IEEE binary format down to float16, and far enough inside float16's range (65504) that it survives a narrowing cast instead of becoming `±np.infty`, which would put `0 * np.infty = np.nan` right back.

## module: `enums`

CLI choice enums. They live here rather than beside the code they select for so that a choice type is not tied to the module that consumes it --- `DatasetName` names an option `ran.data` implements, and `LogLevel` one that `ran.logging_config` does.

## module: `events`

The event data model, in its two representations.

`Populations` is the view that represents the physical sources of events, and `ZXY` that represents the events as they exist in a ML pipeline. Conversion between the two is not strictly lossless. Converting from `ZXY` to populations discards ordering information. Converting from `Populations` to `ZXY` is, however, lossless.

The various event dataclasses take `eq=False` because they hold arrays: a generated `__eq__` compares fields with `==` and a generated `__hash__` hashes them. Both operations raise when called on arrays. They can only be compared for identity.

### `class Events`

A corresponding pair of (particle-level `z`, detector-level `x`) for one set of events. The arrays are row-aligned: row `i` of each is the same event seen at the two levels.

#### Fields

- `z: EventArray`: The particle-level features.
- `x: EventArray`: The detector-level features.

#### Methods

- `concatenate(parts: Sequence[Events]) -> Events`: Concatenate a sequence of events.

### `class Populations`

The physics view of a labelled sample.

`mc` is the simulation, its particle level generation (`mc.z`) paired per event with the corresponding detector level simulation (`mc.x`); that pairing is used to build a response matrix.

`data` is the natural measurement. `truth` is the particle-level answer key. It exists only because every dataset here is a closure test. A real measurement has no such array, and no network may ever see it. Keeping it out of `mc` means a function handed the MCMC cannot reach it. Construct through `create` to set `truth` to `ran.rantypes.constants.TRUTH_SENTINEL`; the field itself is always present.

#### Fields

- `mc: Events`: The simulation.
- `data: EventArray`: The natural measurement.
- `truth: EventArray`: The particle-level answer key.

#### Properties

- `has_truth: bool`: `True` if the sample has a particle-level answer key. `False` if it is populated with `ran.rantypes.constants.TRUTH_SENTINEL`.

Any metric computed against a sentinel `truth` is meaningless but finite, so unfolding code that scores against the particle level has to ask rather than wait to be told.

#### Methods

##### `create(mc: Events, data: EventArray, truth: EventArray | None = None) -> Populations`

Build a sample, filling `truth` with `ran.rantypes.constants.TRUTH_SENTINEL` if there is none.

A real measurement has no answer key. Filling the field rather than dropping it keeps one type for both cases, and keeps the sample trainable: the nature rows of `z` are `truth`, so they reach the generator, and only a finite value there lets `normalize_weights` annihilate them as intended. See `ran.rantypes.constants.TRUTH_SENTINEL` for why not NaN.

`truth` is particle level, so it takes its columns from `mc.z` and its rows from `data`.

**Arguments:**

- `mc: Events`: The simulation.
- `data: EventArray`: The natural measurement.
- `truth: EventArray | None = None`: The particle-level answer key. If `None`, `TRUTH_SENTINEL` is used.

**Returns:**

- `Populations`: The sample.

**Returns:**

- `Populations`: The sample at the new precision.

##### `require_truth() -> EventArray`

Returns `truth` if available or raises a `ValueError` if there is none. Scoring against the sentinel yields a finite, meaningless number instead of an obvious failure, so the particle-level comparisons ask for the answer key through here rather than reading the field.

**Returns:**

- `EventArray`: The particle-level answer key.

##### `interleave() -> ZXY`

Stack into the labelled transport form, nature rows first. The resulting row order is an artifact of stacking rather than anything meaningful, so callers shuffle before splitting.

**Returns:**

- `ZXY`: The stacked events.

### `class ZXY`

Events labelled by provenance: y = 1 for nature, y = 0 for MC. The form in which they get shuffled, split, batched and trained on. `partition` converts to the physics form, `Populations.interleave` back.

#### Fields

- `events: Events`: The events.
- `y: NDArray[np.ubyte]`: The labels.

#### Properties

- `z: EventArray`: The particle-level features.
- `x: EventArray`: The detector-level features.

### `DatasetSplits::select(which: Split = Split.ALL) -> ZXY`

Concatenate the requested splits into one labelled sample. The split a row came from is not recorded on the result: it is a property of the query, not of the events, and nothing downstream of this call can act on it.

**Arguments:**

- `which: Split = Split.ALL`: The splits to concatenate.

**Returns:**

- `ZXY`: The concatenated events.

## module: `results`

### `class UnfoldingPopulations`

The input to an unfolding run: a sample to unfold with, and a sample to score on.

`fit` is train+val and supplies the response (`fit.mc`) and the measurement (`fit.data`). `test` is the held-out split alone, which is where the metrics are computed. The two are disjoint on purpose: fitting on every event and then scoring a subset of them is the convention in the unfolding literature, and it scores an estimator on data it has already seen.

#### Fields

- `fit: Populations`: The sample to unfold with (train+val).
- `test: Populations`: The sample to score on.
