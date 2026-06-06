# RANv4 -- Reweighting Adversarial Networks

An adversarial neural network that learns per-event weights to correct simulated (Monte Carlo) distributions so they match observed data. Built with TensorFlow/Keras.

## Motivation

In particle physics, Monte Carlo (MC) simulations are used to model detector responses and physical processes. These simulations never perfectly reproduce real data -- there are always residual mismodelling effects. Traditional reweighting uses hand-tuned correction factors binned in one or two variables, which scales poorly to high-dimensional feature spaces.

RANv4 replaces this with a learned reweighting: a **generator** network predicts a continuous per-event weight from particle-level (truth) features, while an **adversarial discriminator** tries to distinguish the reweighted simulation from real data. At convergence the discriminator can no longer tell them apart, and the generator's weights constitute an optimal correction.

## How It Works

The system is a two-player adversarial game over event weights:

| Component                | Input                      | Output                                        | Role                                        |
| ------------------------ | -------------------------- | --------------------------------------------- | ------------------------------------------- |
| **Generator** $g(z)$     | Particle-level feature $z$ | Per-event weight (`logplus` = $\log(1+\exp)$) | Predict weights that make MC look like data |
| **Discriminator** $d(x)$ | Detector-level feature $x$ | Data vs MC probability (`sigmoid`)            | Distinguish real data from reweighted MC    |

**Training loop:**

1. **Discriminator step** -- freeze $g$, update $d$ to maximize weighted binary cross-entropy (classify data vs reweighted MC)
2. **Generator step** -- freeze $d$, update $g$ to minimize the same loss (fool the discriminator)
3. Repeat with 5:1 D:G update ratio

Weight normalization ensures the total MC yield is preserved:

$$w_i = \frac{g(z_i)}{\text{mean}(g(z))}$$

In equilibrium, both losses converge to $\log(2)$ and the reweighted MC matches data.

## Installation

Uses [`UV-Astral`](https://docs.astral.sh/uv/) for dependency management. One way to install `uv` is with `pip install uv`. For alternative installation methods, see the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url> && cd RANv4
uv sync
```

### GPU Support

NVIDIA cuDNN is included in the dependencies for GPU acceleration. Ensure compatible CUDA drivers are installed on your system.

## Usage

### Gaussian Datasets

Gaussian datasets are configured via YAML files in `params/`:

```bash
# 1D uncorrelated Gaussian
uv run -m ran --config params/1d_default.yaml

# 2D with correlated covariance
uv run -m ran --config params/2d_correlated.yaml

# 4D and 6D correlated
uv run -m ran --config params/4d_correlated.yaml
uv run -m ran --config params/6d_correlated.yaml

# Customize network and training
uv run -m ran --config params/1d_default.yaml --hidden_units 128 --n_layers 3 --patience 10
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

```bash
# All 6 jet variables
uv run -m ran --dataset jets

# Specific variables
uv run -m ran --dataset jets --variables='("m", "w")'
```

### Other Options

```bash
# Reload an existing run (regenerate plots/metrics)
uv run -m ran --load_run=runs/2026-03-14T061023Z

# SLURM submission
sbatch scripts/submit.sh --config params/2d_correlated.yaml
sbatch scripts/submit.sh --dataset jets
```

| Flag             | Default    | Description                                 |
| ---------------- | ---------- | ------------------------------------------- |
| `--config`       | `None`     | Path to Gaussian YAML config                |
| `--dataset`      | `gaussian` | Dataset type: `gaussian` or `jets`          |
| `--n_samples`    | `500000`   | Number of events per class (data + MC)      |
| `--batch_size`   | `1024`     | Training batch size                         |
| `--hidden_units` | `64`       | Units per hidden layer                      |
| `--n_layers`     | `2`        | Number of hidden layers                     |
| `--patience`     | `5`        | Early stopping patience (epochs)            |
| `--variables`    | all 6      | Jet substructure variables to use           |
| `--load_run`     | `None`     | Path to an existing run directory to reload |

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
uv run -m ran.evaluate

# Evaluate a single run
uv run -m ran.evaluate --run_dir=runs/2026-03-14T061023Z

# Recompute even if metrics.json exists
uv run -m ran.evaluate --force
```

This computes per-dimension 1D Wasserstein distances, Jensen-Shannon divergences, and triangular discriminator (Vincze-LeCam divergence) \[$\times10^3$\] at both detector and particle level, before and after reweighting. Results are saved to `metrics.json` in each run directory.

### Baseline Comparisons

Run [OmniFold](https://github.com/ViniciusMikuni/omnifold) or IBU (Iterative Bayesian Unfolding) on the same datasets for head-to-head comparison:

```bash
# OmniFold — single run
uv run -m ran.baselines.omnifold --run_dir=runs/2026-...

# OmniFold — all runs
uv run -m ran.baselines.omnifold

# Customize OmniFold iterations/epochs
uv run -m ran.baselines.omnifold --run_dir=runs/2026-... --niter=5 --epochs=100

# IBU — single run
uv run -m ran.baselines.ibu --run_dir=runs/2026-...

# IBU — all runs
uv run -m ran.baselines.ibu
```

Results are saved to `metrics_omnifold.json` / `metrics_ibu.json` in each run directory using the same metric format as RAN.

### Leakage Verification

A core correctness requirement is that the generator $g(z)$ never receives $z_\text{true}$ — the particle-level values of real data events, which are unknowable in a real experiment. `scripts/leakage_check.py` verifies this empirically via a **data poisoning test**:

```bash
# Clean run — z_true drawn from N(0, 1) as normal
uv run scripts/leakage_check.py --poison=False

# Poisoned run — z_true overwritten with -999 after x_data is generated
uv run scripts/leakage_check.py --poison=True
```

The poisoned run corrupts every data particle-level value to a nonsense sentinel (-999) while leaving $x_\text{data}$ (the reco-level observations the discriminator actually sees) unchanged. If $g$ had any access to $z_\text{true}$, the poisoned run would produce degraded weights. Both runs should report statistically identical Wasserstein and triangular discriminator improvements. Matching results confirm that no leakage path exists.

## Project Structure

```txt
RANv4/
├── ran/                          Python package
│   ├── __main__.py               Entry point (python -m ran)
│   ├── data/
│   │   ├── config.py             YAML config parsing, sigma promotion
│   │   ├── datasets.py           DatasetSplits, RAN_Dataset, caching
│   │   ├── jets.py               Jet substructure loading and standardization
│   │   └── download.py           One-time Zenodo data download
│   ├── baselines/
│   │   ├── omnifold.py           OmniFold comparison baseline
│   │   └── ibu.py                IBU (Iterative Bayesian Unfolding) baseline
│   ├── models.py                 Generator and discriminator architectures
│   ├── train.py                  Adversarial training loop with early stopping
│   ├── plotting.py               Detector-level, particle-level, and loss curve plots
│   └── evaluate.py               Post-hoc distance metrics (Wasserstein, JS, triangular)
├── params/                       Gaussian config YAML files
│   ├── 1d_default.yaml
│   ├── 2d_correlated.yaml
│   ├── 4d_correlated.yaml
│   └── 6d_correlated.yaml
├── scripts/
│   ├── submit.sh                 SLURM submission script
│   └── leakage_check.py          data poisoning test for z_true leakage
├── tests/                        pytest tests
├── pyproject.toml                Project metadata and dependencies
├── runs/                         Output directory (timestamped subdirectories)
└── .cache/                       Cached datasets
```

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

All configurable via CLI flags or in `ran/train.py`:

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

## Dependencies

- [`TensorFlow`](https://www.tensorflow.org/) >= 2.21
- [`Keras`](https://keras.io/) >= 3.13
- [`NumPy`](https://numpy.org/) >= 2.4
- [`SciPy`](https://scipy.org/) >= 1.15
- [`Matplotlib`](https://matplotlib.org/) >= 3.10
- [`Fire`](https://github.com/google/python-fire) >= 0.7
- [`OmniFold`](https://github.com/ViniciusMikuni/omnifold) >= 0.1 (baseline comparison)
