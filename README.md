# RANv4 -- Reweighting Algorithm Network

An adversarial neural network that learns per-event weights to correct simulated (Monte Carlo) distributions so they match observed data. Built with TensorFlow/Keras.

## Motivation

In particle physics, Monte Carlo (MC) simulations are used to model detector responses and physical processes. These simulations never perfectly reproduce real data -- there are always residual mismodelling effects. Traditional reweighting uses hand-tuned correction factors binned in one or two variables, which scales poorly to high-dimensional feature spaces.

RANv4 replaces this with a learned reweighting: a **generator** network predicts a continuous per-event weight from particle-level (truth) features, while an **adversarial discriminator** tries to distinguish the reweighted simulation from real data. At convergence the discriminator can no longer tell them apart, and the generator's weights constitute an optimal correction.

## How It Works

The system is a two-player adversarial game over event weights:

| Component | Input | Output | Role |
|---|---|---|---|
| **Generator** *g(z)* | Particle-level feature *z* | Per-event weight (softplus) | Predict weights that make MC look like data |
| **Discriminator** *d(x)* | Detector-level feature *x* | Data vs MC probability (sigmoid) | Distinguish real data from reweighted MC |

**Training loop:**

1. **Discriminator step** -- freeze *g*, update *d* to maximize weighted binary cross-entropy (classify data vs reweighted MC)
2. **Generator step** -- freeze *d*, update *g* to minimize the same loss (fool the discriminator)
3. Repeat with 5:1 D:G update ratio

Weight normalization ensures the total MC yield is preserved:

```
w_i = g(z_i) * N_mc / sum_j g(z_j)
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

```bash
# Run with defaults (500k events, batch size 1024, smearing 0.5)
python main.py

# Customize via CLI flags (powered by python-fire)
python main.py --n_samples=1000000 --batch_size=2048 --smearing=0.3
```

| Flag | Default | Description |
|---|---|---|
| `--n_samples` | `500000` | Number of events per class (data + MC) |
| `--batch_size` | `1024` | Training batch size |
| `--smearing` | `0.5` | Gaussian smearing width (detector resolution) |

The pipeline will:
1. Generate (or load from cache) a synthetic Gaussian dataset
2. Split into train / validation / test sets (70 / 10 / 20%)
3. Train the GAN with early stopping (patience = 5 epochs)
4. Save publication-quality PDF plots to `plots/<UTC-timestamp>/`

## Project Structure

```
RANv4/
├── main.py          Entry point and pipeline orchestration
├── models.py        Generator and discriminator architectures
├── train.py         Adversarial training loop with early stopping
├── datasets.py      Synthetic data generation, caching, and splitting
├── plotting.py      Detector-level, particle-level, and loss curve plots
├── pyproject.toml   Project metadata and dependencies
└── plots/           Output directory (timestamped subdirectories)
```

## Synthetic Dataset

The included Gaussian toy dataset demonstrates the reweighting concept:

| | Distribution | Parameters |
|---|---|---|
| **Data (truth)** | Normal | mu=0, sigma=1 |
| **MC (simulation)** | Normal | mu=0.5, sigma=0.9 |

Both are smeared by additive Gaussian noise to simulate detector resolution, producing paired particle-level (*z*) and detector-level (*x*) features. The generator must learn weights on *z* such that the *x* distributions agree.

Datasets are cached to `.cache/` (keyed by parameters) to avoid regeneration across runs.

## Output

Each run produces three plots in `plots/<timestamp>/`:

- **`detector_level.pdf`** -- Histogram of detector-level feature *x* comparing data, unweighted MC, and reweighted MC, with a ratio panel
- **`particle_level.pdf`** -- Same comparison at particle level (*z*): truth vs generated vs reweighted
- **`losses.pdf`** -- Training curves for generator and discriminator (train + validation), with the log(2) equilibrium target

## Training Hyperparameters

Configurable in `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_epochs` | 100 | Maximum training epochs |
| `n_disc_steps` | 5 | Discriminator updates per generator update |
| `lr_g` | 1e-4 | Generator learning rate (Adam) |
| `lr_d` | 1e-4 | Discriminator learning rate (Adam) |
| `patience` | 5 | Early stopping patience (epochs) |
| `min_delta` | 1e-4 | Minimum improvement for early stopping |

## Dependencies

- [TensorFlow](https://www.tensorflow.org/) >= 2.21
- [Keras](https://keras.io/) >= 3.13
- [NumPy](https://numpy.org/) >= 2.4
- [Matplotlib](https://matplotlib.org/) >= 3.10
- [Fire](https://github.com/google/python-fire) >= 0.7
