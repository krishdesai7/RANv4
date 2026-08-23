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

Rebuild a `GaussianConfig` from the `gaussian_params` block of a run's `config.json`, across every format runs/ has ever held - two deprecated formats and one current format. Two share their key names:

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

Test is taken off the end, validation off the end of what remains, so train occupies the front. Only the training split reshuffles between epochs.

##### `generate_gaussian_dataset`

**Arguments**:

- `config_path: Path | None = None` Path to a YAML config file.
- `params: dict | None = None` Dict with keys mu_gen, mu_true, sigma_gen, sigma_true, sigma_detector.
- `n_samples: int = 10**6` Number of samples per class (data and MC).

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Parse the config file or inline params, generate the dataset, and split it into train/val/test splits. Exactly one of config_path or params must be provided.

##### `splits_from_data`

**Arguments**:

- `data: ZXY` One labelled in-memory sample, usually straight from `Populations.interleave()`.

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Shuffle one labelled sample and cut it into train/val/test.

`ZXY` has already checked that the particle- and detector-level arrays are row-aligned and that every label is zero or one, so this does no validation of its own.

## Download

One-time download of jet substructure data from Zenodo (record 3548091).

Downloads Pythia26 and Herwig Z+jets Delphes datasets (17 .npz files each), extracts 6 substructure variables, saves per-variable .npz files to .cache/, and deletes the raw downloads.

### Degenerate jets

Two of the six observables are undefined for a small number of jets. A jet of one constituent has no width, and for $\beta = 1$ the width is $\tau_1$, so $\tau_{21} = \frac{\tau_2}{\tau_1} = \frac00$. This is the case for around $100-300$ jets per array. A jet that soft drop grooms down to a single prong has $m_{\text{sd}} = 0$, so $\ln \rho = \ln \left( \frac{m_{\text{sd}}^2}{p_T^2} \right) = -\infty$. This is the case for a few hundred more jets per array.

`_get_var` computes each observable only where it is defined and fills the rest with a stated value: `_ONE_PRONG_TAU21` (zero, matching <span style="font-variant: small-caps;">OmniFold</span>) and `LOG_RHO_FLOOR` ($-14$, the bottom of the range `JET_OBS` plots $\ln\rho$ over).

The usual alternative is to nudge the denominator or the log argument by an epsilon, and it is worse in three ways.

1. It hides the convention inside a number that reads like a rounding allowance.
2. It depends on the dtype the raw arrays happen to arrive in. For example $10^{-50}$ (used in <span style="font-variant: small-caps;">OmniFold</span>) is below the smallest `float32` denormal, so if arrays were stored as `float32`, it would round away and hand back `NaN` for exactly the jets it was meant to protect.
3. An epsilon scaled to the data, such as $10^{-12} \times \text{mean}(p_T^2)$ (used in <span style="font-variant: small-caps;">OmniFold</span>), is a _different_ epsilon for each of the four arrays, which puts the floor of $\ln\rho$ in a different place for nature than for MC. Several hundred jets per array sit on that floor and thousands more are compressed against it, so the discriminator gets handed a spike whose position differs between the classes for reasons that have nothing to do with physics. The four arrays are two samples that get compared to each other; an observable that means something slightly different in each is not a comparison.

For $\beta = 1$ the jet width is $\tau_1$, so $\tau_{21} = \frac{\tau_2}{\tau_1}$. A jet of one constituent has neither: both vanish and the ratio is 0/0. Zero is what <span style="font-variant: small-caps;">OmniFold</span>'s published results assign it and so is what this code reproduces, but it is a convention rather than a measurement. Zero is also the limit a cleanly two-pronged jet approaches, which a one-constituent jet is obviously not.

### `ran.data.download::_get_var`

Extract a substructure variable from raw arrays.

Two of the six are undefined for a jet the detector or the groomer has left with nothing to measure: $\tau_{21}$ is $\frac00$ when the jet has one constituent, and $\ln\rho = -\infty$ when soft drop leaves no groomed mass. Both are handled by computing the observable only where it exists and filling the rest with a declared value, `_ONE_PRONG_TAU21` and `LOG_RHO_FLOOR`.

**Arguments**:

- `data: dict[str, NDArray]` Raw arrays.
- `var: str` Substructure variable.
- `ptype: str` Particle type.

**Returns**:

- `NDArray[np.double]` Substructure variable array.

### `ran.data.download::_fetch_generator`

Fetch every shard for one generator and concatenate the keys needed. Appends each shard path to `all_raw_paths` so the caller can delete the raw downloads once the per-variable caches have been written.

**Arguments**:

- `gen: str` Generator.
- `cache_dir: Path` Cache directory.
- `progress: Progress` Progress object.
- `all_raw_paths: list[Path]` List of raw paths.

**Returns**:

- `dict[str, NDArray]` Raw arrays.

## Jets

Load jet substructure data for RAN training.

Checks `.cache/` for per-variable `.npz` files. If missing, invokes `download_jet_data` to fetch from Zenodo. Loads, subsamples, z-score standardizes (using MC gen-level statistics only), and builds the train/val/test splits via `RANDataset`.

### `ran.data.jets::load_jet_dataset`

Load jet substructure data and return DatasetSplits.

Each selected substructure variable is z-score standardized using the MC gen-level (`z_gen`) mean and std. The same parameters are applied to all four arrays (`z_true`, `x_data`, `z_gen`, `x_sim`) to avoid information leakage and preserve correlations.

**Arguments**:

- `n_samples: int = 500_000` Number of events to use per class (data and MC).
- `batch_size: int = 1024` Batch size for the returned splits.
- `cache_dir: Path = CACHE_DIR` Directory containing per-variable `.npz` files.
- `variables: frozenset[str] = SUBSTRUCTURE_VARIABLES` Which substructure variables to use.
- `seed: int = 42` Dataset seed, controlling the shuffle, the train/val/test split and the per-epoch batch order. Independent of the weight-init seed passed to `train`.
There is no `dtype` argument. The npz caches on disk are the float64 the Zenodo release ships, and the standardization statistics are computed in that precision; the narrowing to `EVENT_DTYPE` happens once, here, on the way into the pipeline. This is one of the three places data enters and narrows — the others are `_draw_gaussian` and `cubic_sweep.make_particles`.

Narrowing *after* the observables are computed is deliberate, not incidental: `ran.data.download._get_var` upcasts to float64 first, because the ε it uses to protect degenerate jets is below the smallest float32 denormal. Narrow before that and it rounds to zero, handing back `NaN` for exactly the jets the ε exists to protect.

**Returns**:

- `tuple[DatasetSplits, int, dict[str, tuple[T, T]]]` DatasetSplits, feature dimensionality, and standardization parameters {var_name: (mu, sigma)}.
