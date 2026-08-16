# Experiments

## Cubic Sweep

Sweep the cubic detector-distortion strength s and compare RAN vs <span style="font-variant: small-caps;">OmniFold</span>.

For each s in `linspace(0, 20, n_points)`, apply a deterministic non-linear detector response $r(s, z) = z + s \times z^3$ to fixed particle-level samples $z_\text{truth} \sim \mathcal{N}(0,1)$, $z_\text{gen} \sim \mathcal{N}(-1,1)$, unfold `z_gen` back toward `z_truth` with both RAN and <span style="font-variant: small-caps;">OmniFold</span>, and record `Wasserstein(z_truth, z_unfolded)` for each.

RAN and <span style="font-variant: small-caps;">OmniFold</span> run as separate subcommands because they need different Keras backends in different processes (RAN on JAX, <span style="font-variant: small-caps;">OmniFold</span> on TensorFlow -- see `ran/baselines/omnifold.py`). Each writes its own per-point JSON; `collect` joins them on `s_index`. Neither import happens at module scope, so importing this module commits to neither backend.

Usage:

```shell
uv run -m ran sweep ran --s-index 0 --sweep-dir ...
uv run -m ran sweep omnifold --s-index 0 --sweep-dir ...
uv run -m ran sweep collect --sweep-dir ...
```

### `ran.experiments.cubic_sweep::make_particles`

Draw fixed particle-level samples: $z_truth \sim \mathcal{N}(0,1), z_gen \sim \mathcal{N}(-1,1)$.

Deliberately not generic over the float type, unlike everything downstream of it. Those take an array and carry whatever precision they were handed; this one is a source, with no argument to infer a precision from, and `Generator.normal` produces float64 whatever the caller would prefer. A `dtype` parameter could only cast after the fact, which is a narrowing. Narrowing belongs at the boundary that needs it, the way the baselines do it.

#### Arguments

- `n_samples: int`: Number of samples to draw.
- `seed: int = 42`: Random seed.

#### Returns

- `z_truth: NDArray[np.double]`: Particle-level samples from the truth distribution.
- `z_gen: NDArray[np.double]`: Particle-level samples from the generated distribution.

### `ran.experiments.cubic_sweep::_sweep_point`

Resolve one sweep point: its s, the fixed particles, and their response.

The sample comes back in the (n, 1) columns both unfolders take, so neither caller reshapes; `unfolded_wasserstein` ravels them again for scipy.

#### Arguments

- `s_index: int`: Index of the s value to resolve.
- `n_points: int`: Number of s values to resolve.
- `n_samples: int`: Number of samples to draw.
- `seed: int`: Random seed.

#### Returns

- `s: np.double`: The s value.
- `pops: Populations[np.double]`: The populations of the resolved point.

### `ran.experiments.cubic_sweep::run_ran`

Train RAN at one sweep point and write ran\_{index}.json.

Particle samples are drawn once (fixed `seed`) so only s varies across points. `z_unfolded` is the `z_gen` sample reweighted by `g(z_gen)`.

#### Arguments

- `s_index: int`: Index of the s value to resolve.
- `sweep_dir: str | Path`: Directory to write the output JSON to.
- `n_samples: int = 500_000`: Number of samples to draw.
- `n_points: int = 25`: Number of s values to resolve.
- `seed: int = 42`: Random seed.
- `batch_size: int = 1024`: Batch size.
- `ran_epochs: int = 100`: Number of epochs to train for.
- `init_seed: int | None = None`: Weight-initialization seed, drawn from entropy when omitted. Re-run a point with different values to get an ensemble at fixed s; the resolved value is recorded in the output JSON.

#### Returns

- `out: dict`: The output JSON.

### `ran.experiments.cubic_sweep::run_omnifold`

Unfold with OmniFold at one sweep point and write omnifold\_{index}.json.

Draws the same fixed particle samples as run_ran, so the two subcommands
compare like for like. Must run in its own process: the import below pins
Keras to the TensorFlow backend.

#### Arguments

- `s_index: int`: Index of the s value to resolve.
- `sweep_dir: str | Path`: Directory to write the output JSON to.
- `n_samples: int = 500_000`: Number of samples to draw.
- `n_points: int = 25`: Number of s values to resolve.
- `seed: int = 42`: Random seed.
- `omnifold_niter: int = 3`: Number of iterations to run.
- `omnifold_epochs: int = 50`: Number of epochs to train for.
- `omnifold_batch_size: int = 512`: Batch size.

#### Returns

- `out: dict`: The output JSON.

### `ran.experiments.cubic_sweep::_complete_points`

Join the per-method point files, keeping only points both methods finished.

Raises if nothing is complete; warns (but proceeds) when only some points
are, so a partly-failed sweep still produces a plot of what did land.

#### Arguments

- `sweep_dir: Path`: Directory to read the input JSON from.
- `n_points: int`: Number of s values to resolve.

#### Returns

- `complete: list[dict]`: The complete points.

### `ran.experiments.cubic_sweep::collect`

Join the per-method point files into results.npz and a Wasserstein-vs-s PDF.

A point contributes only if both methods wrote it; points where either side
failed are reported as missing rather than silently half-plotted.

#### Arguments

- `sweep_dir: Path`: Directory to read the input JSON from.
- `n_points: int = 25`: Number of s values to resolve.

#### Returns

- `None`: The collected results are written to the sweep directory.
