# CLI Reference

The `deconvolve` CLI is a single Typer command tree with the following subcommands:

- `train`
- `evaluate`
- `report`
- `leakage-check`
- `baseline`
    - `ibu`
    - `omnifold`
- `uncertainty`
    - `freeze`
    - `run`
    - `collect`
- `config show`

Every layerable `train` and `uncertainty` option also resolves as described in [Configuration](configuration.md).

---

## Global Options

`--log-level` (`-L`) is a global option and must be placed **before** the subcommand. For example:

```shell
deconvolve -Ldebug train --config params/1d_default.yaml
```

| Long option | Short | Type | Default | Env Var | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--log-level` | `-L` | `LogLevel` | `info` | `DECONVOLVE_LOG_LEVEL` | Application log level. Options: `debug`, `info`, `warning`, `error`, `critical`. |
| `--install-completion` | | `bool` | | | Install shell autocompletion. |
| `--show-completion` | | `bool` | | | Print the completion script. |
| `--help` | | `bool` | | | |

---

## `deconvolve train`

```shell
deconvolve train [OPTIONS]
```

### Dataset

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-D` | `Dataset` | `gaussian` | Dataset to train on. Options: `gaussian`, `jets`. |
| `--config` | | `Path` | `None` | YAML config file (Gaussian datasets only). |
| `--n-samples` | `-n` | `int` | `500000` | Number of events to generate/load. |
| `--var` | `-v` | `str`, repeatable | all twelve | Jet substructure variable(s) to train on, e.g. `-vm -vw`. Ignored for `gaussian`. |
| `--data-seed` | | `int` | `42` | Seed for dataset sampling and the train/val/test split. |
| `--load-run` | `-r` | `Path` | `None` | Reload a previously saved run directory instead of starting fresh. |

### Architecture & Optimization

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--hidden-units` | `-u` | `int` | `64` | Hidden units per dense layer, generator and discriminator. |
| `--n-layers` | `-l` | `int` | `2` | Number of hidden dense layers, generator and discriminator. |
| `--n-epochs` | `-e` | `int` | `100` | Number of training epochs. |
| `--batch-size` | `-b` | `int` | `1024` | Batch size per training step. |
| `--n-disc-steps` | `-k` | `int` | `5` | Discriminator updates per generator update. |
| `--lr-g` | | `float` | `3e-5` | Generator learning rate (Adam). Tuned; see `benchmarks/README.md`. |
| `--lr-d` | | `float` | `1e-4` | Discriminator learning rate (Adam). |
| `--lambda-dispersion` | | `float` | `0.015` | Penalty on the variance of the generator's normalized weights. `0` disables it. |
| `--seed` | | `int` | `None` (random) | Seed for model initialization and shuffle order. |

### Output

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--run-dir` | | `Path` | `None` (UTC timestamp under `runs/`) | Where to save this run. |
| `--plots` / `--no-plots` | | `bool` | `--plots` | Draw diagnostic figures. Metrics are computed either way. |
| `--log-every` | | `int` | `1` | Log every N epochs. |

There is no `--tag` and no `--runs-dir`; name a run explicitly with `--run-dir`.

---

## `deconvolve evaluate`

Compute distance metrics for one run, or every run under a parent directory (see [Evaluation & Metrics](evaluation.md)).

```shell
deconvolve evaluate [OPTIONS]
```

| Long option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run-dir` | `Path` | `runs` | A single run directory, or a parent directory of several. |
| `--force` / `--no-force` | `bool` | `--no-force` | Recompute even if `metrics.json` already exists. |

There is no `--n-bins`; the histogram resolution used by the Jensen-Shannon and
triangular-discriminator metrics is fixed in code, not exposed on the CLI.

---

## `deconvolve report`

Compile a run directory into one PDF dossier (see [Reporting & Artifacts](reporting.md)).

```shell
deconvolve report RUN_DIR [OPTIONS]
```

| Argument/option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `run_dir` (positional) | `Path` | required | Run directory to report on. |
| `--force` | `bool` | `False` | Rebuild an existing `report.pdf`. |
| `--compile` / `--no-compile` | `bool` | `--compile` | Compile the LaTeX, or stop at `artifacts/report.tex`. |

`run_dir` is a positional argument, not `--run-dir`.

---

## `deconvolve baseline`

Run comparison baselines against the same run directory a `deconvolve train` call produced (see [Comparison Baselines](baselines.md)).

### `deconvolve baseline ibu`

```shell
deconvolve baseline ibu [OPTIONS]
```

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--run-dir` | | `Path` | `runs` | Run directory (or parent of several) to evaluate against. |
| `--force` / `--no-force` | | `bool` | `--no-force` | Recompute even if already evaluated. |
| `--niter` | `-i` | `int` | `10` | IBU iterations. |
| `--purity-threshold` | | `float` | `√0.5 ≈ 0.7071` | Purity threshold used by the response matrix. |

### `deconvolve baseline omnifold`

Runs in a quarantined Python 3.13 subprocess; needs `uv` on `PATH`, and on Perlmutter needs `module load cudatoolkit/12.9` (without it, TensorFlow silently falls back to CPU).

```shell
deconvolve baseline omnifold [OPTIONS]
```

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--run-dir` | | `Path` | `runs` | Run directory (or parent of several) to evaluate against. |
| `--force` / `--no-force` | | `bool` | `--no-force` | Recompute even if already evaluated. |
| `--niter` | `-i` | `int` | `3` | OmniFold iterations. |
| `--n-epochs` | `-e` | `int` | `50` | Epochs per iteration. |
| `--batch-size` | `-b` | `int` | `512` | Batch size. |

---

## `deconvolve leakage-check`

Verifies that `z_true` never reaches a network.

```shell
deconvolve leakage-check [OPTIONS]
```

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--poison` / `--clean` | `-X` / | `bool` | `--clean` | Poison mode injects a sentinel to confirm the check would actually catch a leak. |
| `--sentinel` | `-S` | `float` | `-999.0` | Sentinel value used in `--poison` mode. |
| `--seed` | | `int` | `42` | Model initialization seed. |
| `--init-seed` | | `int` | `0` | Bootstrap/init seed. |

There is no `--config`; this command does not read a dataset config file.

---

## `deconvolve uncertainty`

Bootstrap × seed variance decomposition, run as a `freeze` once, followed by many `run`
cells (typically a SLURM array), then a final `collect`.

### `deconvolve uncertainty freeze`

```shell
deconvolve uncertainty freeze --design-dir DIR [OPTIONS]
```

Resolves the full config stack once and writes `DIR/design.json`, which every
`uncertainty run` cell then reads instead of the ordinary config layers.

| Long option | Short | Type | Default |
| :--- | :--- | :--- | :--- |
| `--design-dir` | `-d` | `Path` | required |
| `--force` | | `bool` | `False` |
| `--n-datasets` | `-B` | `int` | `8` |
| `--n-seeds` | `-S` | `int` | `8` |
| `--n-eval` | | `int` | `100000` |
| `--dataset` | `-D` | `gaussian\|jets` | `jets` |
| `--var` | `-v` | `str`, repeatable | all twelve |
| `--config` | | `Path` | `None` |
| `--batch-size` | `-b` | `int` | `1024` |
| `--n-samples` | `-n` | `int` | `500000` |
| `--hidden-units` | `-u` | `int` | `64` |
| `--n-layers` | `-l` | `int` | `2` |
| `--n-epochs` | `-e` | `int` | `100` |
| `--n-disc-steps` | `-k` | `int` | `5` |
| `--lr-g` | | `float` | `3e-5` |
| `--lr-d` | | `float` | `1e-4` |
| `--lambda-dispersion` | | `float` | `0.015` |
| `--data-seed` | | `int` | `42` |
| `--init-seed` | | `int` | `0` |

### `deconvolve uncertainty run`

```shell
deconvolve uncertainty run --cell N --design-dir DIR
```

Trains one `(bootstrap dataset, init seed)` cell of the design. Takes the same options
as `freeze` plus a required `--cell`/`-c`, but reads their values from the frozen
`design.json` rather than the config stack — an explicit flag on the command line still
overrides the frozen value, nothing else does.

### `deconvolve uncertainty collect`

```shell
deconvolve uncertainty collect --design-dir DIR [OPTIONS]
```

Decomposes a finished design and writes its table, `.npz`, and figure.

| Long option | Short | Type | Default |
| :--- | :--- | :--- | :--- |
| `--design-dir` | `-d` | `Path` | required |
| `--n-datasets` | `-B` | `int` | `8` |
| `--n-seeds` | `-S` | `int` | `8` |
| `--n-bins` | | `int` | `20` |
| `--data-seed` | | `int` | `42` |
| `--init-seed` | | `int` | `0` |

---

## `deconvolve config show`

Prints the resolved value of every setting next to the file, variable, or default it
came from.

```shell
deconvolve config show [COMMAND] [OPTIONS]
```

| Argument/option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `command` (positional, optional) | `str` | `None` | Scope the listing to one command, e.g. `deconvolve config show train`. |
| `--json` | `bool` | `False` | Emit the same content as JSON, for scripting. |
