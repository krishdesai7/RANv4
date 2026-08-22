# Experiments

## Cubic Sweep

Compare RAN against IBU as the detector response gets more non-linear.

For each s in `linspace(0, 20, n_points)`, apply a deterministic non-linear detector response $r(s, z) = z + s \times z^3$ to fixed particle-level samples $z_\text{truth} \sim \mathcal{N}(0,1)$, $z_\text{gen} \sim \mathcal{N}(-1,1)$, unfold `z_gen` back toward `z_truth` with each method, and record `Wasserstein(z_truth, z_unfolded)` for both.

Each point is one `ran sweep ran` invocation writing one `point_NN.json`, so points run concurrently and independently — a point that crashes leaves a gap rather than sinking the sweep. `collect` then reads whatever landed and joins it on `s_index`. Nothing here imports `ran.train` at module scope, so `collect` costs no keras or jax import.

Both methods run in the same invocation, on the same `Populations` object. IBU costs seconds next to 100 epochs of adversarial training, so splitting it into a second pass would buy nothing and would put the two methods on separately-derived samples. Their hyperparameters are pinned at module scope (`_HIDDEN_UNITS`, `_N_LAYERS`, `_PATIENCE`, `_IBU_ITERATIONS`) rather than exposed as flags, because the sweep is a one-variable experiment: only s varies.

### `ran.experiments.cubic_sweep::_ibu_point`

Unfold one sweep point with IBU and score it the way RAN is scored.

IBU narrows to float32 at its own boundary, exactly as it does under `ran baseline ibu` — it has to match the arithmetic its published results were produced with. The weights are returned to float64 before the Wasserstein call, so both arms are *scored* by identical arithmetic and only the unfolding differs.

The fit takes `mc.z`, `mc.x` and `data` — what a real measurement has — and is applied to the same `mc.z` that RAN's weights are applied to. `truth` reaches neither method and appears only in the score.

Past some distortion the purity binning yields fewer than two bins. That is not an error: `weights_for` returns ones and IBU's curve flattens onto the un-unfolded distance. The outcome's `status` records it, and `collect` warns so the plot is read as "IBU gave up here" rather than "IBU did no worse here".

#### Arguments

- `pops: Populations[np.double]`: The populations for one sweep point.

#### Returns

- `ibu_wd: np.double`: Wasserstein distance between `truth` and the IBU-reweighted `mc.z`.
- `outcome: VariableOutcome`: Whether the fit happened, and in how many bins.

Usage:

```shell
ran sweep ran --s-index 0 --sweep-dir ...
ran sweep collect --sweep-dir ...
```

### `ran.experiments.cubic_sweep::make_particles`

Draw fixed particle-level samples: $z_truth \sim \mathcal{N}(0,1), z_gen \sim \mathcal{N}(-1,1)$.

Deliberately not generic over the float type, unlike everything downstream of it. Those take an array and carry whatever precision they were handed; this one is a source, with no argument to infer a precision from, and `Generator.normal` produces float64 whatever the caller would prefer. A `dtype` parameter could only cast after the fact, which is a narrowing. Narrowing belongs at the boundary that needs it, the way the IBU baseline does it.

#### Arguments

- `n_samples: int`: Number of samples to draw.
- `seed: int = 42`: Random seed.

#### Returns

- `z_truth: NDArray[np.double]`: Particle-level samples from the truth distribution.
- `z_gen: NDArray[np.double]`: Particle-level samples from the generated distribution.

### `ran.experiments.cubic_sweep::_sweep_point`

Resolve one sweep point: its s, the fixed particles, and their response.

The sample comes back in the (n, 1) columns the models take, so the caller does not reshape; `unfolded_wasserstein` ravels them again for scipy.

#### Arguments

- `s_index: int`: Index of the s value to resolve.
- `n_points: int`: Number of s values to resolve.
- `n_samples: int`: Number of samples to draw.
- `seed: int`: Random seed.

#### Returns

- `s: np.double`: The s value.
- `pops: Populations[np.double]`: The populations of the resolved point.

### `ran.experiments.cubic_sweep::run_ran`

Run both methods at one sweep point and write point\_{index}.json.

Particle samples are drawn once (fixed `seed`) so only s varies across points. For RAN, `z_unfolded` is the `z_gen` sample reweighted by `g(z_gen)`; for IBU it is the same sample reweighted by the binned correction `_ibu_point` fits. The subcommand keeps the name `ran` because RAN is what it trains — IBU rides along on the populations that training already required.

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

### `ran.experiments.cubic_sweep::_complete_points`

Read the per-point files, keeping only the points that finished.

Raises if nothing is complete; warns (but proceeds) when only some points
are, so a partly-failed sweep still produces a plot of what did land.

#### Arguments

- `sweep_dir: Path`: Directory to read the input JSON from.
- `n_points: int`: Number of s values to resolve.

#### Returns

- `complete: list[dict]`: The complete points.

### `ran.experiments.cubic_sweep::collect`

Join the per-point files into results.npz and a Wasserstein-vs-s PDF.

Points whose run failed are reported as missing rather than silently dropped.

#### Arguments

- `sweep_dir: Path`: Directory to read the input JSON from.
- `n_points: int = 25`: Number of s values to resolve.

#### Returns

- `None`: The collected results are written to the sweep directory.
