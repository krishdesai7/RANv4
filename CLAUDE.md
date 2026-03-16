# RANv4

## Overview

RANv4 is a Reweighting Algorithm Network that uses adversarial learning to unfold (deconvolve) reco-level data to nominal truth.

## Core Algorithm

- `(z_gen, x_sim)` are naturally paired MC events. `(z_true, x_data)` are naturally paired data events.
- **Generator** `g(z)` takes nominal-level events as input and produces per-event weights `w = g(z)`. For data events (y=1), weights are fixed to 1. Weights are normalized to preserve total counts per class.
- These weights are applied to reco-level _distributions_ (not the events themselves). `x_data` is reweighted with w=1 (unchanged), `x_sim` is reweighted with `w = g(z_gen)`.
- **Discriminator** `d(x)` operates at reco level to distinguish these two reweighted distributions.
- **Loss**: weighted BCE — `w_i * y_i * log(d(x_i)) + (1 - y_i) * w_i * log(1 - d(x_i))`.
- **Training**: min-max game. `d` minimizes BCE (correctly distinguish data from reweighted sim). `g` maximizes BCE (generate weights that confound `d`).

## Critical Constraint

**No network should ever have access to `z_true`.** This is the unfolded truth that we do not know in principle. `g` only sees `z` (nominal-level features), never the true particle-level values from data.

## Tooling Preferences

- Prefer `fd` over `find`, `rg` over `grep`, and `fzf` for fuzzy finding. The Rust-based tools are faster and have better defaults. `find`/`grep` are fine as fallbacks.

## Project Structure

```
ran/                   Python package
├── __main__.py        Entry point (python -m ran)
├── data/
│   ├── datasets.py    DatasetSplits, RAN_Dataset, caching
│   ├── jets.py        Jet substructure loading, standardization (JET_OBS, load_jet_dataset)
│   └── download.py    One-time Zenodo download
├── models.py          Generator and discriminator architectures
├── train.py           Adversarial training loop with early stopping
├── plotting.py        Detector-level, particle-level, and loss curve plots
└── evaluate.py        Post-hoc distance metrics (Wasserstein, JS divergence)

submit.sh              SLURM submission script
runs/                  Output directory (timestamped subdirectories)
.cache/                Cached datasets (gaussian .npz, per-variable jet .npz)
```

## Running

```bash
uv run python -m ran                                     # train gaussian (defaults)
uv run python -m ran --dataset jets                      # train on all 6 jet variables
uv run python -m ran --load_run=runs/2026-03-14T061023Z  # reload a saved run
uv run python -m ran.evaluate                            # compute metrics for all runs
uv run python -m ran.evaluate --run_dir=runs/2026-...    # single run
sbatch submit.sh --dataset jets                          # SLURM submission
```

## Tech Stack

- Python >= 3.13, managed with `uv` (no pip)
- TensorFlow / Keras for training
- python-fire for CLI
- Matplotlib for publication-quality plots
- scipy for evaluation metrics (Wasserstein distance, Jensen-Shannon divergence)
