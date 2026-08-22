# RAN: Reweighting Adversarial Networks

`ran` is a library for training and evaluating reweighting adversarial networks (RANs).

Importing anything under `ran` first pins the Keras 3 backend to JAX and enables JAX's 64-bit mode. Both settings are read once, when `jax`/`keras` are first imported, so they must be in place before any submodule imports either.

`ran` defaults to float64 end to end (see :mod:`ran.models`), which JAX silently downcasts to float32 unless x64 mode is on. To trade that precision for GPU throughput,
set `JAX_ENABLE_X64=0` in the environment and switch the `dtype=` arguments in :mod:`ran.models` to `"float32"`.

`setdefault` throughout, so that the environment can be explicitly overridden.

Import the submodule needed (`from ran.workflow import run`); the CLI re-exports below are the sole exception, and they defer their own imports into the command bodies.

## module `evaluate`

Computes distance metrics on test sets for completed runs.

Computes per-dimension 1D Wasserstein distances and Jensen-Shannon divergences, both before and after reweighting. Uses only memory-efficient algorithms: sorted-CDF Wasserstein (O(n log n)) and histogram-based JS divergence.

Usage:

```bash
    ran evaluate                          # all runs in runs/
    ran evaluate --run-dir runs/2026-...  # single run
    ran evaluate --force                  # recompute existing
```

### `apply_to_runs(run_dir: Path, evaluate_one: Callable[[Path], object], description: str, log: Logger) -> None`

Apply `evaluate_one` to a single run directory, or to every run inside one.

A directory is a run if it holds a config.json; otherwise it is treated as a parent directory of runs. In the multi-run case one failure is logged and skipped rather than abandoning the remaining runs.

**Arguments:**

- `run_dir: Path` The directory to evaluate.
- `evaluate_one: Callable[[Path], object]` A function to apply to each run directory.
- `description: str` A description of the evaluation to log.
- `log: Logger` A logger to use for logging.

**Returns:**
None

### `_load_splits(config: dict) -> DatasetSplits`

Load the dataset splits from the config.

**Arguments:**

- `config: dict` The config to load the dataset splits from.

**Returns:**

- `DatasetSplits` The loaded dataset splits.

### `_normalized_histograms(ref: NDArray[T], comp: NDArray[T], weights: NDArray[T] | None = None, n_bins: int = 100) -> tuple[NDArray[np.double], NDArray[np.double]]`

Returns two $(dimensions, n_bins)$ arrays containing the $(p, q)$ probability histograms for each dimension of `ref`/`comp`. Both histograms share one binning per dimension, `n_bins` uniform bins over the combined range, which is what makes the divergence-metrics comparable across dimensions. `weights` reweights `comp` only, and an all-zero histogram is left unnormalized rather than divided by zero.

**Arguments:**

- `ref: NDArray[T]` The reference data.
- `comp: NDArray[T]` The comparison data.
- `weights: NDArray[T] | None` The weights to use for the comparison data.
- `n_bins: int` The number of bins to use for the histograms.

**Returns:**

- `tuple[NDArray[np.double], NDArray[np.double]]` The $(p, q)$ probability histograms for each dimension of `ref`/`comp`

### `_js_per_dim(ref: NDArray[T], comp: NDArray[T], weights: NDArray[T] | None = None, n_bins: int = 100) -> NDArray[np.double]`

Returns JS divergence (squared JS distance) per dimension.

**Arguments:**

- `ref: NDArray[T]` The reference data.
- `comp: NDArray[T]` The comparison data.
- `weights: NDArray[T] | None` The weights to use for the comparison data.
- `n_bins: int` The number of bins to use for the histograms.

**Returns:**

- `NDArray[np.double]` The JS divergence per dimension.

### `_triangular_per_dim(ref: NDArray[T], comp: NDArray[T], weights: NDArray[T] | None = None, n_bins: int = 100) -> NDArray[np.double]`

Triangular discriminator (Vincze-LeCam divergence) per dimension.

$$\Delta(p,q) = \sum_{i=1}^{n} \frac{(p_i - q_i)^2}{p_i + q_i} \times 10^3$$

where $p_i, q_i$ are histogram probability masses. The bin-width factor cancels analytically, so this works directly on normalized histograms.

**Arguments:**

- `ref: NDArray[T]` The reference data.
- `comp: NDArray[T]` The comparison data.
- `weights: NDArray[T] | None` The weights to use for the comparison data.
- `n_bins: int` The number of bins to use for the histograms.

**Returns:**

- `NDArray[np.double]` The triangular discriminator per dimension.

### `evaluate_run(run_dir: Path = RUN_DIR, force: bool = False) -> None`

Compute distance metrics for completed runs.

**Arguments:**

- `run_dir: Path` Path to a single run or a directory containing multiple runs.
- `force: bool` Recompute even if metrics.json already exists.

**Returns:**

- `None`

## module `leakage`

Quick leakage check: poison z_true and verify training is unaffected.

Sets z_true to a silly value in one arm and compares against a clean arm; if any
network can see z_true, the two diverge.

Both arms must use the same `init_seed` or the comparison is meaningless: with
random initialization the run-to-run spread swamps the effect being tested.

## module `plotting`

### `plot_detector_level(test_dataset: ArrayDataset[np.double], g: keras.Model, save_path: Path = Path("plots/detector_level.pdf"), var_info: list[VarInfo] | None = None, ibu_weights: list[NDArray[np.double]] | None = None) -> None:`

Generate detector level plots.

**Arguments:**

- `test_dataset: ArrayDataset[np.double]` The test dataset.
- `g: keras.Model` The generator model.
- `save_path: Path` The path to save the plot to.
- `var_info: list[VarInfo] | None` The per-variable plot config.
- `ibu_weights: list[NDArray[np.double]] | None` The per-variable list of per-event IBU weights for MC events.

**Returns:**

- `None`

### `plot_particle_level(test_dataset: ArrayDataset[np.double], g: keras.Model, save_path: Path = Path("plots/particle_level.pdf"), var_info: list[VarInfo] | None = None, ibu_weights: list[NDArray[np.double]] | None = None) -> None:`

Generate particle level plots.

**Arguments:**

- `test_dataset: ArrayDataset[np.double]` The test dataset.
- `g: keras.Model` The generator model.
- `save_path: Path` The path to save the plot to.
- `var_info: list[VarInfo] | None` The per-variable plot config.
- `ibu_weights: list[NDArray[np.double]] | None` The per-variable list of per-event IBU weights for MC events.

**Returns:**

- `None`

## module `train`

Adversarial training loop for RAN, on Keras 3 with the JAX backend.

The min-max game needs two optimizers driven at different cadences against a shared loss, which does not fit `Model.fit`, so this is a hand-rolled loop. It follows the standard Keras 3 + JAX pattern: model state lives in plain JAX pytrees (never in the `keras.Variable`s) for the duration of training, updates go through `stateless_call`/`stateless_apply`, and each step is a single jitted function. Values are written back into the Keras models at the end so the returned objects are ordinary, saveable `keras.Model`s.

The loss math is plain `jnp`. `stateless_call`/`stateless_apply` are the only Keras calls inside the trace — `lax.scan`, `lax.while_loop` and `jax.random` are all native JAX, so backend-agnostic `keras.ops` bought nothing this module could still use.

### class `TrainResult(NamedTuple)`

All mutable training state, as a JAX pytree.

Held outside the `keras.Model`s so jitted steps stay pure and no host/device sync happens between steps.

**Fields:**

- `g: keras.Model` The generator model.
- `d: keras.Model` The discriminator model.
- `history: dict[str, list[float]]` The training history.
- `seed: int` The random seed used for training.

### def `normalize_weights(raw_w, y)`

Per-event weights: 1 for data, mean-preserving g(z) for MC.

`raw_w` is the raw generator output for every event in the batch. Data events (y=1) are pinned to weight 1; MC events (y=0) are rescaled so their weights sum to the MC event count, preserving the per-class normalization.

The y=1 entries of `raw_w` are multiplied by (1 - y) = 0 in both the sum and the result, so `g`'s output on data rows, which are `z_true`, cannot reach the loss or its gradient. That is what keeps `z_true` out of the model.

**Arguments:**

- `raw_w: Float[Array | np.ndarray, " n"]` The raw generator output for every event in the batch, already squeezed to one dimension.
- `y: Real[Array | np.ndarray, " n"]` The target labels for every event in the batch. `Real` rather than `Float` because the pipeline carries them as `uint8` while the tests pass floats.

**Returns:**

- `Float[Array, " n"]` The normalized weights.

### def `weighted_bce(d_out, y, w)`

Weighted binary cross-entropy.

Reduced with `jnp.sum(...) / n` rather than a mean: for float64 input `keras.ops.mean` picks a float32 compute dtype internally and returns a float64 result carrying ~1e-8 relative error, which would silently undo the float64 policy this project runs on. This module no longer touches `keras.ops`, but the explicit sum-and-divide is what the guard test pins, and anything reaching for `keras.ops` again needs to know. `ops.sum` is unaffected.

**Arguments:**

- `d_out: Float[Array | np.ndarray, " n"]` The discriminator output for every event in the batch, already squeezed to one dimension.
- `y: Real[Array | np.ndarray, " n"]` The target labels for every event in the batch.
- `w: Float[Array | np.ndarray, " n"]` The weights for every event in the batch.

**Returns:**

- `Float[Array, ""]` The scalar loss. Declaring it scalar is what catches a dropped reduction.

**Returns:**

- The weighted binary cross-entropy loss.

### def `_make_steps(g: keras.Model, d: keras.Model, opt_g: keras.optimizers.Optimizer, opt_d: keras.optimizers.Optimizer) -> tuple[JitWrapped, JitWrapped, JitWrapped]`

Build the jitted disc/gen/eval steps, closing over the models.

The models are captured rather than passed so jit sees only array arguments; each returned function is traced once per input shape.

**Arguments:**

- `g` The generator model.
- `d` The discriminator model.
- `opt_g` The generator optimizer.
- `opt_d` The discriminator optimizer.

**Returns:**

- `tuple[JitWrapped, JitWrapped, JitWrapped]` The jitted discriminator, generator, and evaluation steps.

### def `_run_epoch(state: TrainState, train_ds: ArrayDataset, disc_step: JitWrapped, gen_step: JitWrapped, n_disc_steps: int) -> tuple[TrainState, float, float]`

One pass over the training split, returning the new state and mean losses.

`d` updates every batch and `g` every `n_disc_steps`-th batch, i.e. the usual adversarial cadence, giving the discriminator a head start each round. The generator loss is negated back to d's sign convention so the two curves stay directly comparable in the history. Losses are reduced to plain floats so every history series has one element type (`np.mean` would give `np.floating`).

**Arguments:**

- `state: TrainState` The current training state.
- `train_ds: ArrayDataset` The training dataset.
- `disc_step: JitWrapped` The jitted discriminator step.
- `gen_step: JitWrapped` The jitted generator step.
- `n_disc_steps: int` The number of discriminator steps per generator step.

**Returns:**

- `tuple[TrainState, float, float]` The new state and mean losses.

### def `train(splits: DatasetSplits[T], dim: int, n_epochs: int, n_disc_steps: int, lr_g: float, lr_d: float, patience: int, min_delta: float, hidden_units: int, n_layers: int, seed: int | None) -> TrainResult`

Train the generator and discriminator.

This seeds weight initialization _only_. The train/val/test split and the
per-epoch batch order come from the dataset's own seed (`RANDataset`), which draws from an independent generator. Varying `seed` across runs therefore estimates training/initialization variance at fixed data, i.e., the usual HEP model-uncertainty ensemble, while varying the dataset seed instead would fold in split variance.

The networks are Dense-only with no dropout or batch norm and Adam is deterministic, so the two seeds together fully determine a run (up to non-deterministic GPU reductions).

**Arguments:**

- `splits: DatasetSplits[T]` The dataset splits.
- `dim: int` The dimension of the data.
- `n_epochs: int` The number of epochs to train for.
- `n_disc_steps: int` The number of discriminator steps per generator step.
- `lr_g: float` The learning rate for the generator.
- `lr_d: float` The learning rate for the discriminator.
- `patience: int` The number of epochs to wait before early stopping.
- `min_delta: float` The minimum improvement required to continue training.
- `hidden_units: int` The number of hidden units in the networks.
- `n_layers: int` The number of layers in the networks.
- `seed: int | None` The random weight-initialization seed to use for training. `None` draws one from system entropy. Either way the value used is returned, so a run stays reproducible after the fact without having to decide up front that it is worth reproducing.

**Returns:**

- `TrainResult` The training result.

## module `workflow`

### def `_prepare_gaussian[T: np.floating](config: Path | None, saved_config: GaussianConfig | None, batch_size: int, n_samples: int, data_seed: int, *, dtype: _DTypeLike[T]) -> tuple[DatasetSplits[T], int, GaussianConfig]`

Build Gaussian splits from a reloaded run's config, or from a YAML file. Returns the splits, the dimensionality, and the parsed Gaussian params. The last is so a fresh run can record them in its own config.json.

**Arguments:**

- `config: Path | None` The path to the configuration file to use for training.
- `saved_config: GaussianConfig | None` The saved Gaussian configuration.
- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `data_seed: int` The data seed to use for training.
- `dtype: _DTypeLike[T]` Floating type of the generated arrays. Required, and keyword-only: with one call site there is nothing to gain from a default, and without one `T` follows the argument instead of being pinned to it.

**Returns:**

- `tuple[DatasetSplits[T], int, GaussianConfig]` The splits, the dimensionality, and the parsed Gaussian params.

### `def _save_run(g: keras.Model, d: keras.Model, history: dict[str, list[float]], *, batch_size: int, n_samples: int, dim: int, dataset: str, init_seed: int, data_seed: int, gaussian_params: GaussianConfig | None, variables: frozenset[str]) -> Path`

Write models, history and config to a fresh timestamped run directory.

Gaussian params are stored as covariance matrices so runs are self-contained and reloadable without the original YAML. `init_seed` is the resolved weight-init seed, never None, so a run drawn from entropy is still reproducible via --seed after the fact.

**Arguments:**

- `g: keras.Model` The generator model.
- `d: keras.Model` The discriminator model.
- `history: dict[str, list[float]]` The training history.
- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `dim: int` The dimension of the data.
- `dataset: str` The dataset to use for training.
- `init_seed: int` The initial seed to use for training.
- `data_seed: int` The data seed to use for training.
- `gaussian_params: GaussianConfig | None` The Gaussian configuration.
- `variables: frozenset[str]` The variables to use for training.

**Returns:**

- `Path` The path to the run directory.

### `def run(batch_size: int, n_samples: int, config: Path | None, dataset: DatasetName, variables: frozenset[str], load_run: Path | None, hidden_units: int, n_layers: int, patience: int, seed: int | None, data_seed: int) -> None`

Main entry point.

**Arguments:**

- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `config: Path | None` The path to the configuration file to use for training.
- `dataset: DatasetName` The dataset to use for training.
- `variables: frozenset[str]` The variables to use for training.
- `load_run: Path | None` The path to the run to load.
- `hidden_units: int` The number of hidden units in the networks.
- `n_layers: int` The number of layers in the networks.
- `patience: int` The number of epochs to wait before early stopping.
- `seed: int | None` The random weight-initialization seed to use for training. `None` draws one from system entropy. Either way the value used is returned, so a run stays reproducible after the fact without having to decide up front that it is worth reproducing.
- `data_seed: int` The data seed to use for training.

**Returns:**

- `None`
