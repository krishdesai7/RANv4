# Data

## Config

Configuration files for Gaussian datasets are YAML files with the following structure:

```yaml
mu_gen: float |list[float]
mu_true: float | list[float]
sigma_gen: float | list[float] | list[list[float]]
sigma_true: float | list[float] | list[list[float]]
sigma_detector: float | list[float] | list[list[float]]
```

### `ran.data.config::sigma_to_covariance`

Promote `sigma` (scalar, vector, or matrix) to a (dim, dim) covariance matrix, where `dim` is the dimension of the data.

- scalar -> σ²I
- (dim,) vector -> diag(σ²)
- (dim, dim) matrix -> used as-is

Validates positive-definiteness via Cholesky decomposition.

### `ran.data.config::parse_gaussian_config`

Parse a Gaussian YAML config file and return a dictionary with the following keys:

- `dim: int` - The dimension of the data.
- `mu_gen: NDArray[np.double]` The mean of the generated data.
- `mu_true: NDArray[np.double]` The mean of the true data.
- `cov_gen: NDArray[np.double]` The covariance matrix of the generated data.
- `cov_true: NDArray[np.double]` The covariance matrix of the true data.
- `cov_detector: NDArray[np.double]` The covariance matrix of the detector.

### `ran.data.config::gaussian_config_from_run_config`

Rebuild a `GaussianConfig` from the `gaussian_params` block of a run's `config.json`, across every format runs/ has ever held - two depracated formats and one current format. Two share their key names:

- `cov_gen` covariance matrices, written since the type refactor.
- `sigma_gen` (list) covariance matrices under the old name -- master's
  `__main__` stored `cov_gen` as `sigma_gen`.
- `sigma_gen` (scalar) a raw sigma, from before that, needing promotion.

Routing the `sigma_*` spelling through `sigma_to_covariance` resolves the ambiguity without guessing: a raw scalar promotes to σ²I, a vector to diag(σ²), and an already-formed matrix passes through unchanged (checked for shape, symmetry and positive-definiteness on the way). So both readings land on the same covariance, and the `cov_*` spelling needs no promotion at all.

## Two representations

The same events are described two ways, at opposite ends of the pipeline.

`Populations` is the physics form: `mc` (an `Events` pair of generated particle level `mc.z` and simulated detector level `mc.x`, aligned per event), `data` (the measurement), and `truth` (the particle-level answer key). Sources produce it, and analysis consumes it. `truth` sits outside `mc` on purpose, so a function handed the simulation cannot reach the one array no network may see.

`ZXY` is the transport form: an `Events` pair plus a per-event label, `y = 1` for nature and `y = 0` for MC. It is what gets shuffled, split, batched and trained on.

Every dataset here is a closure test, so `truth` is always known; a real measurement is the case where it is not. `Populations.create(mc, data)` covers that by filling `truth` with `TRUTH_SENTINEL`, and `has_truth` distinguishes the two. The stand-in is a number (-2^15) and not NaN because `interleave` puts `truth` into the nature rows of `z`, which the generator forward-passes: `normalize_weights` discards those rows by multiplying by `1 - y = 0`, and that annihilates a number but not a NaN. Metrics computed against a sentinel `truth` are finite and meaningless, so the particle-level comparisons read the answer key through `require_truth()`, which returns it or refuses. `has_truth` is the same question without the exception.

`Populations.interleave()` converts to the transport form (nature rows first, then MC) and `ZXY.partition()` converts back. Only the second composition is lossless: interleaving a partitioned sample discards the shuffled row order. Weight vectors are indexed against a `Populations`, never against a `ZXY`, so nothing should round-trip.

`DatasetSplits.select(Split.TRAIN | Split.VAL)` concatenates the requested splits into one `ZXY`. The split an event came from is a property of the query, not of the event, so it is not recorded on the result.

## Datasets

### `ran.data.datasets::ArrayDataset`

An in-memory `ZXY` with deterministic minibatching.

Iterating yields `({"z": ..., "x": ...}, y)` batches of NumPy arrays. The final batch is short rather than dropped whenever the split length is not a multiple of `batch_size`.

Every split holds a view onto one shared pair of base arrays; slicing is done with fancy indexing at batch time, so splitting costs no extra memory.

#### Fields

- `data: ZXY` The split's events and labels. `data.z` and `data.x` reach through to the underlying `Events`.
- `batch_size: int` Events per batch.
- `shuffle: bool` Re-permute the event order before every pass. Used for the training split; validation and test iterate in fixed order.
- `seed: int` Seed for the reshuffling generator.

#### Properties

- `size: int` Number of events in the dataset.

Each pass draws its permutation from `(seed, pass_index)` rather than from a generator carried across passes, so the order an epoch sees depends only on how many passes preceded it, not on who else has iterated this object.

#### Methods

- `reset() -> None` Return to the first pass.
- `len() -> int` Number of batches per pass.
- `iter() -> Iterator[Batch]` Iterate over the dataset.
- `as_arrays() -> ZXY` Return the whole split in stored order. Callers that just want every event (plotting, metrics, the baselines) should use this instead of concatenating iterations, and usually follow it with `.partition()`.

### `ran.data.datasets::RANDataset`

Dataset class for RAN.

#### Fields

- `batch_size: int` Events per batch.
- `seed: int` Random seed.
- `cache_dir: str | Path` Cache directory.
- `val_fraction: float` Validation fraction.
- `test_fraction: float` Test fraction.

#### Properties

- `dataset: ZXY` The events in shuffled order.
- `splits: DatasetSplits` Dataset splits.

#### Methods

##### `_build_dataset`

**Arguments**:

- `data: ZXY` The events as `interleave` produced them.

**Returns**:

- `ZXY` The same events in shuffled order.

**Description**:

Interleave the nature and MC halves with one fixed-seed permutation.

`interleave` stacks nature (y=1) on MC (y=0); the splits below are contiguous slices, so they would otherwise be single-class. This shuffle happens once and is not repeated per epoch -- it defines the event ordering the splits cut into.

##### `_split_dataset`

**Arguments**:

- `dataset: ZXY` The events in shuffled order.

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Cut the shuffled arrays into contiguous train/val/test splits.

Test is taken off the end, validation off the end of what remains, so train occupies the front -- matching the nested `split_dataset` calls this replaced. Only the training split reshuffles between epochs.

##### `generate_gaussian_dataset`

**Arguments**:

- `config_path: Path | None = None` Path to a YAML config file.
- `params: dict | None = None` Dict with keys mu_gen, mu_true, sigma_gen, sigma_true, sigma_detector.
- `n_samples: int = 10**6` Number of samples per class (data and MC).

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Parse the config file or inline params, generate the dataset, and split it into train/val/test splits Exactly one of config_path or params must be provided.

##### `splits_from_data`

**Arguments**:

- `data: ZXY` One labelled in-memory sample, usually straight from `Populations.interleave()`.

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Shuffle one labelled sample and cut it into train/val/test.

`ZXY` has already checked that the particle- and detector-level arrays are row-aligned and that every label is zero or one, so this does no validation of its own.
