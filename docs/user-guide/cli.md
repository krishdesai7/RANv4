# CLI Reference

The `ran` CLI provides a unified command-line interface for running training, evaluation, reporting, and baseline comparisons.

---

## Global Options

All subcommands accept the following global options:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--log-level`, `-v` | `LogLevel` | `INFO` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `--help` | `bool` | `False` | Show help message and exit. |

---

## `deconvolve train`

Execute the adversarial reweighting training workflow.

```shell
ran train [OPTIONS]
```

### Dataset Options

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `-D` | `gaussian` | Dataset to train on: `gaussian` or `jets`. |
| `--config` | `-c` | `None` | Path to YAML configuration file (required for Gaussian datasets). |
| `--samples` | `-n` | `100000` | Number of events to generate/load. |
| `--variable` | `-v` | `()` | Jet variable name(s) to train on (e.g. `-v m -v w`). Can be specified multiple times. |
| `--data-seed` | | `42` | Seed for dataset sampling and splitting. |

### Network & Training Hyperparameters

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `-e` | `100` | Number of training epochs. |
| `--batch-size` | `-b` | `512` | Batch size per training step. |
| `--hidden-units` | `-u` | `64` | Hidden units per dense layer. |
| `--n-layers` | `-l` | `2` | Number of hidden dense layers for generator and discriminator. |
| `--lr-d` | | `1e-3` | Learning rate for the discriminator (Adam). |
| `--lr-g` | | `1e-3` | Learning rate for the generator (Adam). |
| `--d-steps` | | `5` | Discriminator updates per generator update (D:G ratio). |
| `--seed` | `-s` | `42` | Seed for model initialization and shuffle order. |

### Output Options

| Flag | Description |
| :--- | :--- |
| `--tag` | Optional human-readable tag added to the run directory name. |
| `--runs-dir` | Parent directory for run outputs (default: `runs/`). |

---

## `deconvolve evaluate`

Compute distance metrics on test sets for completed runs.

```shell
ran evaluate [OPTIONS]
```

### Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--run-dir` | `runs/` | Path to a single run directory, or parent directory of multiple runs. |
| `--force` | `False` | Force recomputing metrics even if `metrics.json` already exists. |
| `--n-bins` | `100` | Number of uniform bins per dimension for histograms and distance metrics. |

---

## `deconvolve report`

Generate diagnostic plots and compile the LaTeX report dossier.

```shell
ran report [OPTIONS]
```

### Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--run-dir` | Required | Path to the completed run directory. |
| `--compile-pdf` | `True` | Automatically run `latexmk` / `pdflatex` to produce `report.pdf`. |

---

## `deconvolve baseline`

Run comparison baselines against RAN runs.

### `deconvolve baseline ibu`

Runs Iterative Bayesian Unfolding (IBU):

```shell
ran baseline ibu --run-dir runs/2026-09-19-164500
```

### `deconvolve baseline omnifold`

Runs OmniFold via an isolated TensorFlow worker process:

```shell
ran baseline omnifold --run-dir runs/2026-09-19-164500
```

---

## `deconvolve leakage-check`

Verify that the held-out test split is never observed during training or model selection:

```shell
ran leakage-check --config params/1d_default.yaml
```
