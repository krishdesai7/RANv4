# CLI Reference

The `deconvolve` CLI provides a unified command-line interface for running training, evaluation, reporting, and baseline comparisons.

---

## Global Options

All subcommands accept the following global options:

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--log-level` | `-v` | `LogLevel` | `INFO` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `--help` | | `bool` | `False` | Show help message and exit. |

---

## `deconvolve train`

Execute the adversarial reweighting training workflow.

```shell
deconvolve train [OPTIONS]
```

### Dataset Options

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-D` | `DatasetName` | `gaussian` | Dataset to train on: `gaussian` or `jets`. |
| `--config` | `-c` | `Path` | `None` | Path to YAML config file. |
| `--samples` | `-n` | `int` | `100000` | Number of events to generate/load. |
| `--variable` | `-v` | `str` | All | Jet var(s) to train on (e.g. `-vm -vw`). |
| `--data-seed` | | `int` | `42` | Seed for dataset sampling and splitting. |

### Network & Training Hyperparameters

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--epochs` | `-e` | `int` | `100` | Number of training epochs. |
| `--batch-size` | `-b` | `int` | `512` | Batch size per training step. |
| `--hidden-units` | `-u` | `int` | `64` | Hidden units per dense layer. |
| `--n-layers` | `-l` | `int` | `2` | Number of hidden dense layers for generator and discriminator. |
| `--lr-d` | | `float` | `1e-3` | Learning rate for the discriminator (Adam). |
| `--lr-g` | | `float` | `1e-3` | Learning rate for the generator (Adam). |
| `--d-steps` | | `int` | `5` | Discriminator updates per generator update (D:G ratio). |
| `--seed` | `-s` | `int` | `42` | Seed for model initialization and shuffle order. |

### Output Options

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--tag` | | `string` | `None` | Optional human-readable tag added to the run directory name. |
| `--runs-dir` | | `Path` | `runs/` | Parent directory for run outputs. |

---

## `deconvolve evaluate`

Compute distance metrics on test sets for completed runs.

```shell
deconvolve evaluate [OPTIONS]
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
deconvolve report [OPTIONS]
```

### ReportOptions

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
deconvolve baseline ibu --run-dir runs/2026-09-19-164500
```

### `deconvolve baseline omnifold`

Runs OmniFold via an isolated TensorFlow worker process:

```shell
deconvolve baseline omnifold --run-dir runs/2026-09-19-164500
```

---

## `deconvolve leakage-check`

Verify that the held-out test split is never observed during training or model selection:

```shell
deconvolve leakage-check --config params/1d_default.yaml
```
