# RAN: Reweighting Adversarial Networks

An adversarial neural network that learns per-event weights to correct simulated (Monte Carlo) distributions so they match observed data. Built with Keras 3 on the JAX backend.

## Motivation

In particle physics, Monte Carlo (MC) simulations are used to model detector responses and physical processes. These simulations never perfectly reproduce real data. There are always residual mismodelling effects. Traditional reweighting uses hand-tuned correction factors binned in one or two variables, which scales poorly to high-dimensional feature spaces.

RANv4 replaces this with a learned reweighting: a **generator** network predicts a continuous per-event weight from particle-level (truth) features, while an **adversarial discriminator** tries to distinguish the reweighted simulation from real data. At convergence the discriminator can no longer tell them apart, and the generator's weights constitute an optimal correction.

## Model

The system is a two-player adversarial game over event weights:

| Component                | Input                      | Output                                         | Role                                          |
| ------------------------ | -------------------------- | ---------------------------------------------- | --------------------------------------------- |
| **Generator** $g(z)$     | Particle-level feature $z$ | Per-event weight (`logplus` = $\log(1+\exp)$ ) | Predict weights that make MC look like nature |
| **Discriminator** $d(x)$ | Detector-level feature $x$ | Data vs MC probability (`sigmoid`)             | Distinguish real data from reweighted MC      |

### Training loop

1. **Discriminator step**: freeze $g$, update $d$ to maximize weighted binary cross-entropy (classify Data vs reweighted Simulation)
2. **Generator step**: freeze $d$, update $g$ to minimize the same loss (fool the discriminator).
3. Repeat with 5:1 D:G update ratio

Weight normalization ensures the total MC yield is preserved:

$$w_i = \frac{g(z_i)}{\text{mean}(g(z))}$$

In equilibrium, both losses converge to $\log(2)$ and the reweighted MC matches data.

## Installation

Requires Python >= 3.13. Uses [`uv`](https://docs.astral.sh/uv/) for dependency management. One way to install it is with `pip install uv`; for alternatives see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

```shell
git clone https://github.com/krishdesai7/RANv4.git
cd RANv4
uv sync
```

This installs the `ran` console script into `.venv/bin`. Commands below are
written as `ran ...`; from a checkout without an activated virtualenv, prefix
them with `uv run` (`uv run ran train --config params/1d_default.yaml`).
Tab completion for subcommands, flags and enum values is available with:

```shell
ran --install-completion
```

### GPU Support

The JAX dependency is platform-resolved:

#### Linux, x86_64

Built against `jax[cuda13]` on x86_64 Linux, compiled against CUDA version 13.0. For NVIDIA GPUs, the CUDA 13 runtime libraries are available as pypi wheels that the JAX binary is built against, so only a compatible NVIDIA driver is needed.

#### macOS, arm64 (Apple Silicon)

The official macOS arm64 wheels for JAX do not provide GPU acceleration. Therefore JAX and consequentially RAN only offer CPU support on Apple Silicon;

Experimental alternatives, such as `jax-mps` or `IREE`-based workflows, may enable Metal acceleration, but these configurations are not tested or supported by RAN. Users should independently validate their correctness and performance.

## Usage

### Gaussian Datasets

Gaussian datasets are configured via YAML files. Examples are provided in `params/`:

```shell
# 1D uncorrelated Gaussian
ran train --config params/1d_default.yaml

# 2D with correlated covariance
ran train --config params/2d_correlated.yaml

# 4D and 6D correlated
ran train --config params/4d_correlated.yaml
ran train --config params/6d_correlated.yaml

# Customize network and training
ran train --config params/1d_default.yaml --hidden-units 128 --n-layers 3 --patience 10
```

YAML config format (see `params/` for examples):

```yaml
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9 # scalar, vector, or full covariance matrix
sigma_true: 1.0
sigma_detector: 0.5
```

Sigma values are promoted to covariance matrices:

- scalar $\to \sigma^2 I$
- vector $\to \text{diag}(\sigma^2)$
- matrix $\to$ as-is

### Jet Substructure

```shell
# All 6 jet variables
ran train --dataset jets

# Specific variables
ran train --dataset jets --variable m --variable w
```

### Other Options

```shell
# Reload an existing run (regenerate plots/metrics)
ran train --load-run runs/2026-03-14T061023Z

# Enable debug logging for any command
ran --log-level DEBUG train --config params/1d_default.yaml

# SLURM submission
sbatch scripts/submit.sh --config params/2d_correlated.yaml
sbatch scripts/submit.sh --dataset jets
```

| Flag             | Default        | Description                                           |
| ---------------- | -------------- | ----------------------------------------------------- |
| `--config`       | `None`         | Path to Gaussian YAML config                          |
| `--dataset`      | `gaussian`     | Dataset type: `gaussian` or `jets`                    |
| `--n-samples`    | `500_000`      | Number of events per class (data + MC)                |
| `--batch-size`   | `1024`         | Training batch size                                   |
| `--hidden-units` | `64`           | Units per hidden layer                                |
| `--n-layers`     | `2`            | Number of hidden layers                               |
| `--patience`     | `5`            | Early stopping patience (epochs)                      |
| `--variable`     | all 6          | Repeat once for each jet substructure variable to use |
| `--load-run`     | `None`         | Path to an existing run directory to reload           |
| `--seed`         | system entropy | Weight-initialization seed (see [Seeding](#seeding))  |
| `--data-seed`    | `42`           | Data generation, shuffle, split and batch order       |

The pipeline will:

1. Generate (or load from cache) the dataset
2. Split into train / validation / test sets (70 / 10 / 20%)
3. Train the RAN with early stopping
4. Save models, training history, and plots to `runs/<UTC-timestamp>/`
5. Compute distance metrics on the test set

### Evaluation

Distance metrics can be computed independently on existing runs:

```bash
# Evaluate all runs
ran evaluate

# Evaluate a single run
ran evaluate --run-dir runs/2026-03-14T061023Z

# Recompute even if metrics.json exists
ran evaluate --force
```

This computes per-dimension 1D Wasserstein distances, Jensen-Shannon divergences, and triangular discriminator (Vincze-LeCam divergence) \[$\times10^3$\] at both detector and particle level, before and after reweighting. Results are saved to `metrics.json` in each run directory.

### Baseline Comparisons

Run [OmniFold](https://github.com/ViniciusMikuni/omnifold) or IBU (Iterative Bayesian Unfolding) on the same datasets for head-to-head comparison:

```bash
# OmniFold — single run
ran baseline omnifold --run-dir runs/2026-03-14T061023Z

# OmniFold — all runs
ran baseline omnifold

# Customize OmniFold iterations/epochs
ran baseline omnifold --run-dir runs/2026-03-14T061023Z --niter 5 --epochs 100

# IBU — single run
ran baseline ibu --run-dir runs/2026-03-14T061023Z

# IBU — all runs
ran baseline ibu
```

Results are saved to `metrics_omnifold.json` / `metrics_ibu.json` in each run directory using the same metric format as RAN.

### Cubic-Response Sweep

Run each step as its own process so RAN's JAX backend and OmniFold's TensorFlow backend never share an interpreter:

```bash
ran sweep ran --s-index 0 --sweep-dir runs/cubic-sweep
ran sweep omnifold --s-index 0 --sweep-dir runs/cubic-sweep
ran sweep collect --sweep-dir runs/cubic-sweep
```

### Leakage Verification

A core correctness requirement is that the generator $g(z)$ never receives $z_\text{true}$, the particle-level values of measured data events, which are unknowable in a real experiment. The `leakage-check` command verifies this empirically via a **data poisoning test**:

```bash
# Clean run — z_true drawn from N(0, 1) as normal
ran leakage-check --clean

# Poisoned run — z_true overwritten with -999 after x_data is generated
ran leakage-check --poison
```

The poisoned run corrupts every data particle-level value to a nonsense sentinel (-999) while leaving $x_\text{data}$ (the reco-level observations the discriminator actually sees) unchanged. If $g$ had any access to $z_\text{true}$, the poisoned run would produce degraded weights. Both runs should report statistically identical Wasserstein and triangular discriminator improvements. Matching results confirm that no leakage path exists.

Both arms must share `--seed`, or initialization variance swamps the effect and
the arms differ even with no leakage. With it fixed, detector-level results are
bit-identical between the clean and poisoned arms.

## Backend

`src/ran/__init__.py` sets `KERAS_BACKEND=jax` and `JAX_ENABLE_X64=1`. **Any `ran.*` import must come before `import keras`.** The backend is fixed at the
first keras import. `src/ran/train.py` raises a clear error if the backend has been incorrectly initialized.

The project is run in float64 precision end to end. Improved GPU throughput can be achieved by reducing precision to float32 by setting `JAX_ENABLE_X64=0` and switching the`dtype=`arguments in`src/ran/models.py`.

`src/ran/train.py` is a hand-rolled loop, since the two-optimizer min-max game does not fit a standard `keras.Model.fit`. It does, however, follow the standard Keras 3 + JAX pattern:

- Model state lives in JAX pytrees (`TrainState`) for the duration of training
- Updates are applied through `stateless_call`/`stateless_apply`
- Each step is a single jitted function.
- Values are written back into the Keras models at the end, so the returned objects are ordinary saveable `keras.Model`s.
- Loss math is written in backend-agnostic `keras.ops`.
- Only the gradient transform and `jit` are native JAX.

It is important to flag two potentially unexpected behaviours that may lead to bugs:

- **`keras.ops.mean` is not float64-safe.**
  - For float64 input, it selects a float32 compute dtype internally and returns a float64 result carrying ~1e-8 relative error.
  - This is why RAN's `src/ran/train.py` reduces with `ops.sum(...) / n` instead of `ops.mean(...)`;
  - `tests/test_train.py` guards this behaviour.
  - `ops.sum` is unaffected.
- **`omnifold` cannot run on JAX.**
  - Its `weighted_binary_crossentropy` calls raw `tf.gather`, which raises `TracerArrayConversionError` under JAX tracing.
  - Therefore, `src/ran/baselines/omnifold.py` pins `KERAS_BACKEND=tensorflow` at import and must be the entry point of its own process.
  - This is why RAN does not allow `omnifold` to be imported from a module that has already touched JAX, and why the cubic sweep splits into separate `sweep ran` / `sweep omnifold` subcommands.

## Seeding

Two independent randomness axes, deliberately kept separate:

| Seed          | Controls                                               |
| ------------- | ------------------------------------------------------ |
| `--data-seed` | Generation, shuffle, train/val/test split, batch order |
| `--seed`      | Weight initialization only                             |

`--seed` defaults to a draw from system entropy, and the value used is recorded in `config.json`, so a run stays reproducible after the fact.

Configs predating this default used `data_seed=42`.

To estimate model uncertainty, ensemble, i.e. rerun on the same inputs with fresh initializations and take the variance as the model uncertainty, is a loop over `--seed` at fixed `--data-seed`.

Because the networks are Dense-only (no dropout or batch norm) and Adam is deterministic, the two seeds together fully determine a run, up to non-deterministic GPU reductions.

Force bitwise reproducibility with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. This costs throughput and is not needed for variance estimates.

## Project Structure

```txt
RANv4/
├── src/ran/                      Python package
│   ├── __init__.py               Pins KERAS_BACKEND=jax and JAX_ENABLE_X64=1
│   ├── __main__.py               Fallback entry point (python -m ran)
│   ├── cli.py                    Unified Typer command tree; target of the `ran` script
│   ├── workflow.py               Training and reload workflow
│   ├── logging_config.py         Structured application logging
│   ├── leakage.py                Data-poisoning leakage check
│   ├── py.typed                  PEP 561 typing marker
│   ├── rantypes/
│   │   ├── events.py             Split, Events, ZXY, Populations, DatasetSplits
│   │   ├── configs.py            GaussianConfig, RunConfig
│   │   ├── results.py            UnfoldingPopulations, VariableOutcome, IBUResult
│   │   ├── constants.py          Zenodo record, cache layout, jet plot metadata
│   │   ├── enums.py              CLI choice enums
│   │   └── types.py              TypedDicts and array aliases
│   ├── data/
│   │   ├── config.py             YAML config parsing, sigma promotion
│   │   ├── datasets.py           DatasetSplits, RANDataset, caching
│   │   ├── jets.py               Jet substructure loading and standardization
│   │   └── download.py           One-time Zenodo data download
│   ├── baselines/
│   │   ├── _shared.py            Run config and populations shared by both (keras-free)
│   │   ├── omnifold.py           OmniFold comparison baseline (pins TensorFlow)
│   │   └── ibu.py                IBU (Iterative Bayesian Unfolding) baseline
│   ├── experiments/
│   │   └── cubic_sweep.py        Cubic-response RAN-vs-OmniFold sweep
│   ├── models.py                 Generator and discriminator architectures
│   ├── train.py                  JAX adversarial training loop with early stopping
│   ├── plotting.py               Detector-level, particle-level, and loss curve plots
│   └── evaluate.py               Post-hoc distance metrics (Wasserstein, JS, triangular)
├── params/                       Gaussian config YAML files
│   ├── 1d_default.yaml
│   ├── 2d_correlated.yaml
│   ├── 4d_correlated.yaml
│   └── 6d_correlated.yaml
├── scripts/
│   ├── submit.sh                 Training and baseline SLURM submission script
│   └── submit_sweep.sh           Packed cubic-response sweep launcher
├── tests/                        pytest tests
├── .github/workflows/ci.yml      Lint, format, types, complexity, tests, audit
├── Justfile                      Development recipes (just check, just fix, ...)
├── pyproject.toml                Project metadata and dependencies
├── runs/                         Output directory (timestamped subdirectories)
└── .cache/                       Cached datasets
```

`src/ran/data/` and `src/ran/baselines/` carry their own `README.md` with
module-level detail.

## Datasets

### Gaussian (Synthetic)

Configurable multivariate Gaussian distributions with correlated covariance matrices. Supports arbitrary dimensionality and correlation structure via YAML config files. Both truth and MC samples are smeared by additive Gaussian noise to simulate detector resolution, producing paired particle-level ($z$) and detector-level ($x$) features.

### Jet Substructure (Physics)

`Herwig` (data) vs `Pythia26` (MC) $Z+$ jets at high $p_T$ (200 GeV), with [`Delphes`](https://github.com/delphes/delphes) detector simulation. Automatically downloaded from from [Zenodo record 3548091](https://zenodo.org/record/3548091) if not already present in `.cache/`.

| Variable | Symbol         | Description                   |
| -------- | -------------- | ----------------------------- |
| `m`      | $m/\text{GeV}$ | Jet mass                      |
| `M`      | $M$            | Jet constituent multiplicity  |
| `w`      | $w$            | Jet width                     |
| `tau21`  | $\tau_{21}$    | N-subjettiness ratio          |
| `zg`     | $z_g$          | Groomed jet momentum fraction |
| `sdm`    | $\ln\rho$      | Log soft-drop jet mass        |

All variables are z-score standardized using MC gen-level statistics only (no information leakage).

## Output

Each run produces a timestamped directory under `runs/` containing:

- **`generator.keras`**/**`discriminator.keras`** -- Saved model checkpoints
- **`history.npz`** -- Training loss history
- **`config.json`** -- Run configuration (reproducibility)
- **`detector_level.pdf`** -- Histogram comparing data, MC, and reweighted MC at detector level with ratio panel
- **`particle_level.pdf`** -- Same comparison at particle level
- **`losses.pdf`** -- Training curves with log(2) equilibrium target
- **`metrics.json`** -- Wasserstein, JS divergence, and triangular discriminator (before/after)
- **`metrics_omnifold.json`** -- Same metrics from OmniFold baseline (if run)
- **`metrics_ibu.json`** -- Same metrics from IBU baseline (if run)

## Training Hyperparameters

These are internal training defaults in `src/ran/train.py`; the CLI-exposed
training options are listed above.

| Parameter      | Default | Description                                |
| -------------- | ------- | ------------------------------------------ |
| `n_epochs`     | 100     | Maximum training epochs                    |
| `n_disc_steps` | 5       | Discriminator updates per generator update |
| `lr_g`         | 1e-4    | Generator learning rate (Adam)             |
| `lr_d`         | 1e-4    | Discriminator learning rate (Adam)         |
| `patience`     | 5       | Early stopping patience (epochs)           |
| `min_delta`    | 1e-4    | Minimum improvement for early stopping     |
| `hidden_units` | 64      | Units per hidden layer                     |
| `n_layers`     | 2       | Number of hidden layers                    |

## Development

```shell
just check     # all local, read-only validation (format, lint, types, complexity, tests)
just fix       # apply safe lint fixes, then format
just test      # pytest, forwards extra args
just typecheck # pyrefly
just ci        # the full CI suite
just           # list every recipe
```

GitHub Actions runs the same suite on push. Lint and format are
[`ruff`](https://docs.astral.sh/ruff/), type checking is
[`pyrefly`](https://pyrefly.org/) at `--min-severity info`, and
[`complexipy`](https://github.com/rohaquinlop/complexipy) enforces a maximum
cognitive complexity of 10.

## Dependencies

- [`JAX`](https://docs.jax.dev/) >= 0.11 \(`jax[cuda13]` on x86_64 Linux\)
- [`Keras`](https://keras.io/) >= 3.15.1
- [`NumPy`](https://numpy.org/) >= 2.5.1
- [`SciPy`](https://scipy.org/) >= 1.18.0
- [`Matplotlib`](https://matplotlib.org/) >= 3.11.1
- [`Typer`](https://typer.tiangolo.com/) >= 0.27.1
- [`PyYAML`](https://pyyaml.org/) >= 6.0.3
- [`OmniFold`](https://github.com/ViniciusMikuni/omnifold) >= 0.1.36
