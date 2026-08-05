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

## Datasets

### `ran.data.datasets::ArrayDataset`

In-memory (z, x, y) arrays with deterministic minibatching.

Iterating yields `({"z": ..., "x": ...}, y)` batches of NumPy arrays. The final batch is short rather than dropped whenever the split length is not a multiple of `batch_size`.

Every split holds a view onto one shared pair of base arrays; slicing is done with fancy indexing at batch time, so splitting costs no extra memory.

#### Fields

- `z: NDArray[np.double]` Particle-level features, shape (n_events, dim).
- `x: NDArray[np.double]` Detector-level features, shape (n_events, dim).
- `y: NDArray[np.ubyte]` Per-event class label (1 = data, 0 = MC), shape (n_events,).
- `batch_size: int` Events per batch.
- `shuffle: bool` Re-permute the event order before every pass. Used for the training split; validation and test iterate in fixed order.
- `seed: int` Seed for the reshuffling generator.

#### Properties

- `n_events: int` Number of events in the dataset.

Each pass draws its permutation from `(seed, pass_index)` rather than from a generator carried across passes, so the order an epoch sees depends only on how many passes preceded it, not on who else has iterated this object.

#### Methods

- `reset() -> None` Return to the first pass.
- `len() -> int` Number of batches per pass.
- `iter() -> Iterator[Batch]` Iterate over the dataset.
- `as_arrays() -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]` Return the dataset as NumPy arrays. Flattened (z, x, y) arrays, in stored order. Callers that just want every event (plotting, metrics, the baselines) should use this instead of concatenating iterations.

### `ran.data.datasets::_parse_params`

Normalize inline params dict into the shape `parse_gaussian_config` returns.
**Arguments**:

- `params: dict[str, ArrayLike]` Inline params dict.

**Returns**:

- `dict[str, int | NDArray[np.double]]` Normalized params dict.

### `ran.data.datasets::RANDataset`

Dataset class for RAN.

#### Fields

- `batch_size: int` Events per batch.
- `seed: int` Random seed.
- `cache_dir: str | Path` Cache directory.
- `val_fraction: float` Validation fraction.
- `test_fraction: float` Test fraction.

#### Properties

- `dataset: tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]` Dataset arrays in shuffled order.
- `splits: DatasetSplits` Dataset splits.

#### Methods

##### `_build_dataset`

**Arguments**:

- `z: NDArray[np.double]` Particle-level features, shape (n_events, dim).
- `x: NDArray[np.double]` Detector-level features, shape (n_events, dim).
- `y: NDArray[np.ubyte]` Per-event class label (1 = data, 0 = MC), shape (n_events,).

**Returns**:

- `tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]` Shuffled dataset arrays.

**Description**:

Interleave the data and MC halves with one fixed-seed permutation.

The arrays arrive as data (y=1) stacked on MC (y=0); the splits below are contiguous slices, so they would otherwise be single-class. This shuffle happens once and is not repeated per epoch -- it defines the event ordering the splits cut into.

##### `_split_dataset`

**Arguments**:

- `dataset: tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]` Shuffled dataset arrays.

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

##### `splits_from_arrays`

**Arguments**:

- `z: NDArray[np.double]` Particle-level features, shape (n_events, dim).
- `x: NDArray[np.double]` Detector-level features, shape (n_events, dim).
- `y: NDArray[np.ubyte]` Per-event class label (1 = data, 0 = MC), shape (n_events,).

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Build train/val/test splits directly from in-memory (z, x, y) arrays.

z (particle level) and x (detector level) must have matching first dimension; y is the per-event class label (1 = data, 0 = MC).
