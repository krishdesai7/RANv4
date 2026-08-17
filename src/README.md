# RAN: Reweighting Adversarial Networks

`ran` is a library for training and evaluating reweighting adversarial networks (RANs).

Importing anything under `ran` first pins the Keras 3 backend to JAX and enables JAX's 64-bit mode. Both settings are read once, when `jax`/`keras` are first imported, so they must be in place before any submodule imports either.

`ran` defaults to float64 end to end (see :mod:`ran.models`), which JAX silently downcasts to float32 unless x64 mode is on. To trade that precision for GPU throughput,
set `JAX_ENABLE_X64=0` in the environment and switch the `dtype=` arguments in :mod:`ran.models` to `"float32"`.

`setdefault` throughout, so that the environment can be explicitly overridden. That is how :mod:`ran.baselines.omnifold` pins itself back to TensorFlow.

Importing this package must not import `keras`. :mod:`ran.baselines.omnifold` hard-sets the backend to TensorFlow at import, and :mod:`ran.train` refuses to load on anything but JAX, so a package `__init__` that pulled in both would make the two mutually unimportable, and would leak `KERAS_BACKEND=tensorflow` into every subprocess besides.

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
