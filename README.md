# RANv4 -- Reweighting Algorithm Network

An adversarial neural network that learns per-event weights to correct simulated (Monte Carlo) distributions so they match observed data. Built with TensorFlow/Keras.

## Motivation

In particle physics, Monte Carlo (MC) simulations are used to model detector responses and physical processes. These simulations never perfectly reproduce real data -- there are always residual mismodelling effects. Traditional reweighting uses hand-tuned correction factors binned in one or two variables, which scales poorly to high-dimensional feature spaces.

RANv4 replaces this with a learned reweighting: a **generator** network predicts a continuous per-event weight from particle-level (truth) features, while an **adversarial discriminator** tries to distinguish the reweighted simulation from real data. At convergence the discriminator can no longer tell them apart, and the generator's weights constitute an optimal correction.

## How It Works

The system is a two-player adversarial game over event weights:

| Component                | Input                      | Output                           | Role                                        |
| ------------------------ | -------------------------- | -------------------------------- | ------------------------------------------- |
| **Generator** _g(z)_     | Particle-level feature _z_ | Per-event weight (softplus)      | Predict weights that make MC look like data |
| **Discriminator** _d(x)_ | Detector-level feature _x_ | Data vs MC probability (sigmoid) | Distinguish real data from reweighted MC    |

**Training loop:**

1. **Discriminator step** -- freeze _g_, update _d_ to maximize weighted binary cross-entropy (classify data vs reweighted MC)
2. **Generator step** -- freeze _d_, update _g_ to minimize the same loss (fool the discriminator)
3. Repeat with 5:1 D:G update ratio

Weight normalization ensures the total MC yield is preserved:

```
w_i = g(z_i) / mean(g(z))
```

At Nash equilibrium both losses converge to log(2) and the reweighted MC matches data.

## Installation

Requires **Python >= 3.13**. Uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

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
sigma_gen: 0.9        # scalar, vector, or full covariance matrix
sigma_true: 1.0
sigma_detector: 0.5
```

Sigma values are promoted to covariance matrices: scalar -> sigma^2 * I, vector -> diag(sigma^2), matrix -> used as-is.

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
sbatch submit.sh --config params/2d_correlated.yaml
sbatch submit.sh --dataset jets
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
3. Train the GAN with early stopping
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

This computes per-dimension 1D Wasserstein distances, Jensen-Shannon divergences, and triangular discriminator (Vincze-LeCam divergence, x10^3) at both detector and particle level, before and after reweighting. Results are saved to `metrics.json` in each run directory.

### OmniFold Baseline Comparison

Run [OmniFold](https://github.com/ViniciusMikuni/omnifold) on the same datasets for head-to-head comparison:

```bash
# Single run
uv run -m ran.omnifold_baseline --run_dir=runs/2026-...

# All runs
uv run -m ran.omnifold_baseline

# Customize OmniFold iterations/epochs
uv run -m ran.omnifold_baseline --run_dir=runs/2026-... --niter=5 --epochs=100
```

Results are saved to `metrics_omnifold.json` in each run directory using the same metric format as RAN.

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
│   ├── models.py                 Generator and discriminator architectures
│   ├── train.py                  Adversarial training loop with early stopping
│   ├── plotting.py               Detector-level, particle-level, and loss curve plots
│   ├── evaluate.py               Post-hoc distance metrics (Wasserstein, JS, triangular)
│   └── omnifold_baseline.py      OmniFold comparison baseline
├── params/                       Gaussian config YAML files
│   ├── 1d_default.yaml
│   ├── 2d_correlated.yaml
│   ├── 4d_correlated.yaml
│   └── 6d_correlated.yaml
├── scripts/
│   └── leakage_check.py          z_true leakage sanity check
├── submit.sh                     SLURM submission script
├── pyproject.toml                Project metadata and dependencies
├── runs/                         Output directory (timestamped subdirectories)
└── .cache/                       Cached datasets
```

## Datasets

### Gaussian (Synthetic)

Configurable multivariate Gaussian distributions with correlated covariance matrices. Supports arbitrary dimensionality and correlation structure via YAML config files. Both truth and MC samples are smeared by additive Gaussian noise to simulate detector resolution, producing paired particle-level (_z_) and detector-level (_x_) features.

### Jet Substructure (Physics)

Herwig (data) vs Pythia26 (MC) Z+jets at high pT (200 GeV), with Delphes detector simulation. Downloaded from [Zenodo record 3548091](https://zenodo.org/record/3548091).

| Variable | Symbol      | Description                   |
| -------- | ----------- | ----------------------------- |
| `m`      | _m_ \[GeV\] | Jet mass                      |
| `M`      | _M_         | Jet constituent multiplicity  |
| `w`      | _w_         | Jet width                     |
| `tau21`  | tau_21      | N-subjettiness ratio          |
| `zg`     | z_g         | Groomed jet momentum fraction |
| `sdm`    | ln(rho)     | Log soft-drop jet mass        |

All variables are z-score standardized using MC gen-level statistics only (no information leakage).

## Output

Each run produces a timestamped directory under `runs/` containing:

- **`generator.keras`** / **`discriminator.keras`** -- Saved model checkpoints
- **`history.npz`** -- Training loss history
- **`config.json`** -- Run configuration (reproducibility)
- **`detector_level.pdf`** -- Histogram comparing data, MC, and reweighted MC at detector level with ratio panel
- **`particle_level.pdf`** -- Same comparison at particle level
- **`losses.pdf`** -- Training curves with log(2) equilibrium target
- **`metrics.json`** -- Wasserstein, JS divergence, and triangular discriminator (before/after)
- **`metrics_omnifold.json`** -- Same metrics from OmniFold baseline (if run)

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

- [TensorFlow](https://www.tensorflow.org/) >= 2.21
- [Keras](https://keras.io/) >= 3.13
- [NumPy](https://numpy.org/) >= 2.4
- [SciPy](https://scipy.org/) >= 1.15
- [Matplotlib](https://matplotlib.org/) >= 3.10
- [Fire](https://github.com/google/python-fire) >= 0.7
- [OmniFold](https://github.com/ViniciusMikuni/omnifold) >= 0.1 (baseline comparison)
