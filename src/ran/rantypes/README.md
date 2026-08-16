# RAN Types

Records, constants and aliases shared across the package.

These live apart from the code that uses them because a process gets a single Keras backend, fixed at the first `keras` import, so `ran.cli` and `ran.baselines._shared` have to stay importable without committing to one. They cannot reach into `ran.train` (JAX) or `ran.baselines.omnifold` (TensorFlow) for a shared declaration, so the
declaration must live here instead. Nothing in this package imports keras or jax
at runtime.

Types owned by exactly one module stay with that module. E.g., `TrainResult` and `TrainState` are in `ran.train`.

## module: `ran.rantypes.configs`

Validated views of a run's Gaussian parameter set and `config.json` that together configure a run.

### `class RunConfig`

A validated view of a run's `config.json`. `source` is the raw dict, kept because `_load_splits` reconstructs the dataset from it and must see exactly what the run recorded.

#### Fields

- `source: dict[str, Any]`: The raw config.json.
- `dataset: Literal["gaussian", "jets"]`: The dataset to use.
- `dim: int`: The dimension of the Gaussian parameter set.
- `n_samples: int`: The number of samples to use.
- `batch_size: int`: The batch size to use.
- `data_seed: int`: The seed for the dataset.
- `variable_names: tuple[str, ...]`: The names of the jet variables to use.

## module: `ran.rantypes.constants`

Fixed values: the Zenodo jet dataset, its cache layout, plot metadata, the default purity threshold, and the stand-in for an absent particle level.

### Constant: `LOG_RHO_FLOOR: Final[float]`

Soft drop grooms some jets down to a single prong, leaving $m_{sd} = 0$ and $\ln(\rho) = \ln\left(\frac{m_{sd}^2}{p_T^2}\right) = -\infty$. Those jets take this value instead, which is both the bottom of the plotted range below and the reason it can be: no jet with a groomed mass reaches it.

Furthermore, unlike $-\infty$ (or the $10^{-100}$ the upstream <span style="font-variant: small-caps;">OmniFold</span> observable adds inside the log, which floors at $-230$) it does not swamp the mean and variance that the features are standardized by. Degenerate jets land in the underflow bin, where they belong.

### Constant: `TRUTH_SENTINEL: Final[np.double]`

Stands in for a particle-level value that does not exist, in the one place that happens: a real measurement, which has data but no answer key.

It has to be finite. `normalize_weights` annihilates the nature rows of the generator's output by multiplying by $(1 - y) = 0$, and under IEEE 754 that annihilates any finite number but not a NaN. `0 * np.nan` is `np.nan`, which then spreads through the class-normalizing sum to every weight in the batch, and through `jax.grad` to every gradient. Sanitizing the masked output would not help either, since the `np.nan` is already in `z` when `g` forward-passes it. The fix has to sit at the input, and be an ordinary number.

$-2^{15}$ is that number: absurd on sight for any standardized observable, exact in every IEEE binary format down to float16, and far enough inside float16's range (65504) that it survives a narrowing cast instead of becoming `±np.infty`, which would put `0 * np.infty = np.nan` right back.
