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

## Data Representations

The same events carry three shapes. Two are defined in
`src/ran/rantypes/events.py` and sit at opposite ends of the host pipeline; the
third, in `src/ran/data/device.py`, is what training actually runs on.

`Populations` is the physics form and holds three things: `mc`, an `Events` pair
of generated particle level (`mc.z`) and simulated detector level (`mc.x`)
aligned per event; `data`, the measurement; and `truth`, the particle-level
answer key. `truth` is deliberately _not_ inside an `Events` alongside `data`,
so a function handed the simulation cannot reach `z_true` — the constraint
above, expressed in the type rather than by convention.

A real measurement has no `truth`, so `Populations.create` fills it with
`TRUTH_SENTINEL` (-2^15, exactly representable in every IEEE binary format,
which is what lets `has_truth` compare by equality at any precision — see
Precision); `has_truth` reports the difference and
`require_truth()` refuses, which is how the particle-level metrics read the
answer key. The stand-in
has to be an ordinary number rather than NaN: `interleave` puts `truth` in the
nature rows of `z`, where `normalize_weights` annihilates the generator's
output by multiplying by `1 - y = 0`. That masks a number but not a NaN, which
would otherwise reach every weight in the batch and every gradient.

`ZXY` is the transport form: an `Events` pair plus a label, `y = 1` for nature
and `y = 0` for MC. It is what shuffles and splits.

Sources build a `Populations` and call `interleave()`; analysis calls
`partition()` on the way back out. Only `partition(interleave(...))` is
lossless — the reverse discards the shuffled row order, and weight vectors are
indexed against a `Populations`, so nothing should round-trip.
`DatasetSplits.select(Split.TRAIN | Split.VAL)` draws one `ZXY` from any
combination of splits.

`TrainSplit`/`EvalSplit` are the training form, and unlike the other two they
are device-resident `jax.Array`s registered as JAX pytrees.
`DeviceSplits.from_splits(splits)` is the single host→device transfer of a run;
after it, no batch ever crosses the boundary again. The train split stays flat
and is gathered by index inside the scan, so XLA fuses the gather into the first
`Dense`; the eval splits are pre-batched to a uniform shape and padded, with a
`mask` field that is 1 for a real row and 0 for filler. The mask enters every
sum in `normalize_weights` and `bce_sums`, so a padded batch reports exactly the
number an unpadded one would.

Keeping the first two forms on host NumPy is deliberate: they feed SciPy,
Matplotlib, npz I/O and the IBU baseline, none of which want device arrays.
Only `src/ran/data/device.py` is device-resident.

## Tooling Preferences

- Prefer `fd` over `find`, `rg` over `grep`, and `fzf` for fuzzy finding. The Rust-based tools are faster and have better defaults. `find`/`grep` are fine as fallbacks.
- When working with Python, you may invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Project Structure

The package lives under `src/` and is importable as `ran`.

```text
src/ran/                      Python package
├── __init__.py               Pins KERAS_BACKEND=jax, JAX_ENABLE_X64=0 (see Backend)
├── __main__.py               Fallback entry point (python -m ran)
├── cli.py                    Unified Typer command tree; `ran` script targets cli:app
├── workflow.py               Training and reload workflow behind `ran train`
├── report.py                 PDF dossier behind `ran report` (see Reporting)
├── leakage.py                Data-poisoning leakage check behind `ran leakage-check`
├── logging_config.py         Rich structured application logging
├── py.typed                  PEP 561 marker
├── rantypes/
│   ├── events.py             Split, Events, ZXY, Populations, DatasetSplits
│   ├── configs.py            GaussianConfig, RunConfig, REQUIRED_KEYS
│   ├── results.py            UnfoldingPopulations, VariableOutcome, IBUResult
│   ├── constants.py          Zenodo record, cache layout, JET_OBS
│   ├── enums.py              LogLevel, DatasetName (CLI choice enums)
│   └── types.py              TypedDicts and array aliases (annotation-space only)
├── data/
│   ├── config.py             YAML config parsing, sigma promotion, gaussian_config_from_run_config
│   ├── datasets.py           ArrayDataset (host container), DatasetSplits, RANDataset, caching
│   ├── jets.py               Jet substructure loading, standardization (JET_OBS, load_jet_dataset)
│   ├── device.py             Device-resident training form (TrainSplit/EvalSplit, batch order)
│   └── download.py           One-time Zenodo download
├── baselines/
│   ├── _shared.py            Run config + populations a baseline needs, minus the unfolder
│   ├── ibu.py                IBU (Iterative Bayesian Unfolding) baseline
│   ├── omnifold.py           OmniFold baseline, host half (see OmniFold)
│   └── _omnifold_worker.py   PEP 723 script; 3.13 + TensorFlow, never imported
├── uncertainty/
│   ├── design.py             Bootstrap x seed grid: resampling, one cell, loading
│   ├── variance.py           Two-way ANOVA components, covariances, quantile binning
│   └── report.py             Decomposition table, variance.npz, correlation.pdf
├── models.py                 Generator and discriminator architectures
├── train.py                  Fused JAX training program (owns TrainResult/TrainState/RunCarry)
├── plotting.py               Detector-level, particle-level, and loss curve plots
├── templates/
│   └── report.tex            LaTeX skeleton `report.py` fills in; see Reporting
├── timing.py                 Optional per-phase wall clock (see Timing)
└── evaluate.py               Post-hoc distance metrics (Wasserstein, JS, triangular discriminator)

params/                       Gaussian config YAML files
├── 1d_default.yaml
├── 2d_correlated.yaml
├── 4d_correlated.yaml
└── 6d_correlated.yaml

scripts/
├── submit.zsh                 SLURM submission script
├── submit_omnifold.zsh        OmniFold against an existing run dir (see OmniFold)
├── submit_hparam.zsh          Packed hyperparameter arm sweep (paired on seed)
├── submit_precision.zsh       float32 vs float64 paired ensemble
└── submit_uncertainty.zsh     Packed bootstrap x seed grid (see Uncertainty)

tests/                        pytest tests (572 cases; `just test`, or `just test-fast`)
Justfile                      Dev recipes: just validate / lint-fix / test / type-check / ci
.github/workflows/ci.yml      Same suite on push
runs/<timestamp>Z/            One run. Two files at the top, the rest below:
├── config.json               Every knob that produced the run
├── report.pdf                `ran report` output; the thing a human reads
└── artifacts/                Everything else, flat -- see Reporting
    ├── generator.keras, discriminator.keras, params.npz, history.npz
    ├── metrics.json, metrics_ibu.json, ibu_outcomes.json, ibu_weights.npz
    ├── metrics_omnifold.json, omnifold_weights.npz
    ├── timings.json          Merged across passes (see Timing)
    ├── detector_level.pdf, particle_level.pdf, losses.pdf, selection.pdf
    └── report.tex            The filled-in template, kept for debugging
.cache/                       Regenerable cache; relocatable via RAN_CACHE_DIR (see Caching)
├── gaussian_*.npz            Generated Gaussian datasets, keyed on the promoted covariances
├── mass.npz, mult.npz, ...   Per-variable jet caches from the Zenodo download
└── jax/                      XLA persistent compilation cache
```

`src/ran/rantypes/`, `src/ran/data/`, `src/ran/baselines/` and
`src/ran/uncertainty/` each carry their own `README.md`.

## Running

The package installs a `ran` console script (`[project.scripts]` →
`ran.cli:app`), which is the canonical entry point. In a checkout, prefix it
with `uv run` to use the project environment without activating it
(`uv run ran train ...`); `python -m ran` still works via `__main__.py`. Shell
completion comes from `ran --install-completion` and needs the script name, so
it does not work through `python -m`.

One Typer command tree. Flags are kebab-case; subcommands are
`train`, `evaluate`, `report`, `leakage-check`, `baseline {ibu,omnifold}`,
`uncertainty {run,collect}`. `--log-level` is global and
goes before the subcommand.

Every knob that changes a run is reachable from `ran train` and recorded in
`config.json` — architecture (`-u`, `-l`), optimization (`--lr-g`, `--lr-d`,
`-k`/`--n-disc-steps`, `--lambda-dispersion`) and the loop (`-e`/`--n-epochs`)
and both seeds. `--lr-g` defaults to 3e-5, measured rather than chosen: see
"What tuning actually found" in `benchmarks/README.md`. `--lambda-dispersion`
penalizes the variance of `g`'s normalized MC weights and defaults to **0.015**,
also measured rather than chosen: it is the same axis `lr_g` acts on
indirectly, pushed directly, and "The dispersion penalty: the trade made
explicit" in `benchmarks/README.md` is the paired 12-observable sweep that
picked the value. It is shipped, not under test — a run left at the default
has the penalty **on**, which is the setting any baseline should be compared
against.
Model selection is not a flag: it is fixed to the detector-level MMD argmin
(see Training Loop). `--no-plots` skips the figures, which are a large share of
a short run's wall clock and no part of scoring one; metrics still run, and
`--load-run` on the same directory draws them afterwards.

`--run-dir` names where a run saves, and a sweep needs it. The default is a UTC
timestamp at second resolution, which several runs of identical shape launched
together will collide on; `_new_run_dir` refuses a directory already holding a
`config.json` and disambiguates the default rather than overwriting silently.
An empty directory is accepted, so a launcher can create one to redirect logs
into before training starts.

```bash
ran train --config params/1d_default.yaml                     # 1D uncorrelated
ran train --config params/1d_default.yaml --seed 7            # reproducible init (see Seeding)
ran train --config params/2d_correlated.yaml                  # 2D with covariance
ran train --dataset jets                                      # train on all 6 jet variables
ran train --dataset jets --var m --var w                      # a subset of jet variables
ran train --dataset jets --lr-g 3e-4 -k 2 --no-plots          # tuning: see Hyperparameters
ran train --dataset jets --seed 3 --run-dir runs/hp_x/lrg1e-4_seed03  # one arm of a sweep
ran train --load-run runs/2026-03-14T061023Z                  # reload a saved run
ran evaluate                                                  # compute metrics for all runs
ran evaluate --run-dir runs/2026-...                          # single run
ran baseline ibu --run-dir runs/2026-...                      # IBU comparison
ran baseline omnifold --run-dir runs/2026-...                 # OmniFold (see OmniFold)
ran report runs/2026-...                                      # PDF dossier (see Reporting)
ran leakage-check --clean                                     # z_true leakage sanity check
ran --log-level DEBUG train --config params/1d_default.yaml
sbatch scripts/submit.zsh                                      # end-to-end 6-var jet run
sbatch scripts/submit.zsh --dataset gaussian --config params/2d_correlated.yaml
bash scripts/submit_hparam.zsh                                 # hyperparameter arms, 3 levels x 8 seeds
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_...     # paired comparison of the arms
```

Development recipes go through `just` (`just` alone lists them):

```bash
just validate   # format, lint, type-check, complexity, tests -- all read-only
just lint-fix   # safe lint fixes, then format
just test -k train   # extra args forward to pytest
just test-fast  # the same suite minus `slow`, for a check mid-work
```

**`just test-fast` deselects `@pytest.mark.slow` and is the only thing that
skips anything.** `just test`, `just validate` and CI all run the whole suite.
The split is there because the cost is wildly uneven: 37 of the 572 cases are
~55s of a ~76s run, and the other ~500 are ~23s together, so a quick pass
costs a third of the time and gives up a fixed, known list rather than a
random one.

Nothing in the suite runs OmniFold. Its worker needs a TensorFlow
environment that cannot exist here, so `tests/test_omnifold.py`
substitutes a stub worker over the same `.npz` contract and tests the
seam instead --- see OmniFold.

The marker goes on a test for a *reason*, not for a measured duration --- a
stopwatch threshold rots as the hardware and the suite move. A test is `slow`
if it **runs a training program** (one `train()` call is ~0.5s even with the
XLA cache warm), **shells out to `pdflatex`**, or **averages many random draws
to measure a statistical property** (`tests/test_mmd_floor.py`). Write a new
test against the piece directly and it costs a few milliseconds and needs no
marker; reach for a full run and it costs a hundred times that and does.

`scripts/submit.zsh` is the full pipeline rather than a bare `ran train`: it
trains, runs the IBU baseline on the same run directory, reloads once so the
figures come back out with the baseline overlaid (`workflow.run` picks up
`ibu_weights.npz` only if it exists when the plots are drawn), recomputes
metrics, then `module load texlive` and `ran report` to leave a PDF at the top
of the run directory. It defaults to the full **twelve**-observable jet run at
`-n 1600000 -l 3 -u 128`, with `RAN_TIMING=1` exported so the run reports
where its wall clock went. Extra flags reach `ran train`; those defaults
are prepended, and click keeps the last occurrence of a scalar option, so
anything on the command line still wins.

That last rule is why the script does **not** name the twelve observables as
`--var` flags, and instead lets `load_jet_dataset`'s own default stand.
`--var` is repeatable, so click *appends* rather than replacing: naming all
twelve would turn `sbatch scripts/submit.zsh --var m` into thirteen names with a
duplicate, which `load_jet_dataset` rejects. Left off, a subset stays
selectable from the command line.

`-n` is resolved against the cache rather than hardcoded. `load_jet_dataset`
raises rather than truncating when asked for more than is on disk, which at
1.6M — a number chosen to sit at the edge of the release — would burn the whole
allocation on an immediate `ValueError`. The script reads the real count off
the cache and clamps, reporting when it does; a cold cache falls through to the
requested number.

There is no `--patience` flag, and none is needed: `n_epochs` (100 by
default -- `train`'s own default, and `workflow.run` does not pass it, so
there is no CLI flag for it either) is a fixed `scan` trip count, and the
best state is restored **always**, selected on the host after the run rather
than during it. At the shipped 1.6M samples and 100 epochs that is ~21.8k
generator and ~109k discriminator updates (the train split is 70% of
`n_samples`, batched at 1024 and grouped by `-k`; it was ~13.6k / ~68k at the
earlier 1M).

The job asks for **`--qos=shared --time=00:15:00`** on a quarter node
(`--gpus=1 --cpus-per-task=32`, and deliberately no `--mem`) — nothing shards
across devices, so three of a node's four A100s would sit idle, and `shared`
charges for the
quarter actually used while backfilling into gaps a whole-node request cannot
reach. `-c 32` is mandatory rather than a preference: the `gpu_shared` queue
requires exactly 32 logical cores per GPU. Do not add `--mem` — the scheduler
converts a memory request into an equivalent core count and enforces the larger
of the two, so `--mem=64G` reads as a 38-core request and the queue rejects it
(`requires you to request 32.0 cores per GPU, job requested 38 cores (adjusted
for memory)`). Omitted, memory comes out proportional to the cores at ~54GB,
against a run that needs ~105MB on device and under a gigabyte on host.

`-C gpu` is the 40GB A100 (~1200 nodes). `-C gpu&hbm80g` gets the 80GB part
from a pool of ~200 — more queue time for headroom this job does not use.

The wall clock is sized from
`benchmarks/boundary.py` on an A100 (4.6s compile, 0.034s per epoch at the same
500k x 6D shape): training is ~15s at the parameters above and the pipeline is
minutes, dominated by npz loading and matplotlib. **Warm the jet cache on a
login node first** — a cold cache pulls 3.3GB from Zenodo (Pythia26 1.55GB +
Herwig 1.75GB) inside the job and will exceed the `debug` ceiling:

```bash
uv run python -c "from ran.data import load_jet_dataset; load_jet_dataset(n_samples=1000)"
```

The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset`
raises if `n_samples` exceeds what is on disk. 1.6M is therefore a request
against the ceiling rather than a safe round number, which is why the script
clamps it to what the cache actually holds instead of asserting a figure.

The cubic-response sweep (`ran sweep`, `src/ran/experiments/`,
`scripts/submit_sweep.zsh`) has been retired and sits under `legacy/`, which is
a holding pen and not a supported path: it is not importable as `ran`, not
covered by `just test`, and slated for deletion. Nothing in the package
references it.

## Releasing

One button. **Actions -> Release -> Run workflow**, pick a bump (or leave it
blank), and the job does the rest: rewrite the version, write the changelog
stanza, re-lock, commit to master, tag, and publish the GitHub release.

Nothing is released by merging a PR. `ran evaluate` and friends do not read the
version, so a release is a labelling act, not a build step -- it exists to give
a result you can cite a fixed point in the code.

**The bump comes from PR labels.** rooster reads every PR merged since the last
tag and takes the largest bump any label implies: `breaking` -> minor, anything
else -> patch. Labels in `ignore-labels` (`internal`, `ci`, `testing`,
`automations`) contribute nothing, so a release consisting only of plumbing
aborts with "No pull requests found after applying ignored labels" -- working as
intended, not a failure to debug. Give a PR a real label if you want it to show
up in the changelog. The `bump` input overrides the inference when you want to
force one.

**Versions stay below 1.0.** `major-labels` is empty and the workflow refuses to
tag anything outside `0.x`, so no label and no merge can walk the project into
1.0. That number is reserved for the first PyPI publication; see below.

**master is protected by a ruleset, and the release job is the one exception.**
Every human change to master goes through a PR with a green `ci`. Force-pushes
and deletions are blocked. The release job pushes directly because a
write-scoped deploy key is the ruleset's bypass actor, and `actions/checkout`
loads it from the `RELEASE_SSH_KEY` secret. `GITHUB_TOKEN` cannot be given a
bypass here: GitHub only accepts the GitHub Actions app as a bypass actor on
organization-owned repositories, and this repository belongs to a user. A PAT
would work but would carry a person's identity, which would hand that person
direct push access to master as a side effect.

**PyPI publishing is off.** The `publish_pypi` input defaults to false, so a
release never depends on credentials. Before it can ever succeed, two things
must change: `PYPI_TOKEN` has to exist in repository secrets, and the
distribution has to be renamed -- `ran` is already taken on PyPI by an unrelated
package, so uploading under that name returns 403 regardless of the token.
`ranv4` is free.

## OmniFold

`ran baseline omnifold` is the second comparison baseline, and the only part of
this repository that does not run in this repository's environment. Three facts
make that necessary: OmniFold needs TensorFlow, TensorFlow publishes no wheels
for Python 3.14, and Keras binds its backend once per interpreter. All three
are **intra-interpreter** constraints, so all three dissolve at a process
boundary.

`src/ran/baselines/_omnifold_worker.py` carries a PEP 723 header pinning
`requires-python = "==3.13.*"` plus `omnifold` and `tensorflow`, and
`uv run --no-project` provisions exactly that, in an interpreter that cannot
import `ran`. The two halves exchange one `.npz` file. `--no-project` is
load-bearing: without it uv resolves the script against this project, whose
`>=3.14` floor cannot be reconciled with the worker's pin.

The worker is inside the package but is not part of it. Nothing imports it and
nothing may: its module-level `KERAS_BACKEND=tensorflow` would race the
package's `jax` pin. Three mechanisms keep it that way, all enforced rather than
documented --- `pyproject.toml` pins `[tool.ruff.per-file-target-version]` for
`**/*_worker.py` to `py313`, pyrefly excludes the same glob, and
`tests/test_omnifold.py::TestQuarantine` asserts the module is absent from
`sys.modules` and that TensorFlow is not importable at all.

**The ruff pin is not hygiene.** Ruff infers `py314` from `requires-python`, and
its formatter rewrites `except (A, B):` into PEP 758's unparenthesized form ---
a `SyntaxError` on 3.13, which killed the worker at import the first time it was
formatted. This is the mirror image of the `timing.py` note under Tech Stack,
where the same syntax is deliberate. A test compiles the worker to catch a
regression.

**`uv` must be on `PATH` at runtime**, since it is what provisions the worker.
Its absence is translated into a readable message rather than a
`FileNotFoundError` from inside `subprocess`, because the fix is an install.

**The worker environment is not in `uv.lock`.** uv resolves the PEP 723 header
on first use, which needs outbound network, and compute nodes generally have
none. Warm it on a login node, the way the jet cache is warmed:

```bash
uv run --no-project src/ran/baselines/_omnifold_worker.py
```

### It runs on the CPU, silently, without a CUDA 12 toolkit

**On Perlmutter `module load cudatoolkit/12.9` is mandatory.** The default
environment leads `LD_LIBRARY_PATH` with four CUDA **13.2** trees and the
`tensorflow` wheel is a CUDA **12** build; exactly one library goes unreachable,
`libcusolver.so.11`, and one is enough for TF to skip registering every GPU. It
then runs on the CPU and **raises nothing** --- the weights come back correct,
tens of times slower, and the baseline looks like it worked.

So the worker reports the device it used and `_warn_if_on_cpu` warns when it was
not a GPU. That warning is the only signal this failure produces; do not silence
it. `benchmarks/gpu_coexistence.py` measures the whole thing and its README
section records the numbers.

What that benchmark also settled: a TensorFlow subprocess gets the GPU **even
with JAX's default 75% preallocation held by the parent**. The worker peaks at
1.07GB against the 9.4GB that survives, so no `XLA_PYTHON_CLIENT_*` tuning is
needed. That was the risk worth checking before any of this was written, and it
did not bind.

### Its own job, not a step in `submit.zsh`

`scripts/submit.zsh` does not run OmniFold. That job asks for
`--time=00:15:00`, and OmniFold alone measured **~41 minutes** on the shipped
configuration --- 1.6M samples, twelve observables, `niter=3`, 50 epochs --- so
it would not fit in what is left after RAN trains. It gets
`scripts/submit_omnifold.zsh` instead, which takes an existing run directory and
asks for 75 minutes:

```zsh
sbatch scripts/submit_omnifold.zsh runs/<timestamp>Z
```

That script loads `cudatoolkit/12.9`, runs the baseline, unloads it, redraws the
figures, re-scores and rebuilds the report. **The module unload is an EXIT trap,
not zsh's `{ } always { }`** --- `always` does not run under `set -e`, which
ERR_EXIT leaves before reaching, so a failed unfolding would have left the CUDA
12 toolkit loaded over whatever ran next in the allocation. Measured, not
assumed.

### Getting OmniFold onto the figures

Presence is the mechanism, and it is the same one IBU has always used.
`_load_baseline_weights` returns one `BaselineOverlay` per `*_weights.npz` that
exists in `artifacts/` when the figures are drawn, so:

```zsh
ran baseline omnifold --run-dir runs/<timestamp>Z   # writes omnifold_weights.npz
ran train --load-run runs/<timestamp>Z              # reloads, redraws with it
```

`--load-run` reloads the saved generator rather than training, so the redraw is
cheap and the run is untouched. There is no separate "add OmniFold to the plots"
command because there is nothing for it to do that `--load-run` does not.

`plotting.BaselineOverlay` is what made a second baseline cheap. The overlay
used to be a bare `ibu_weights: list[EventArray] | None` threaded through six
functions; with two baselines that would have become two parameters in six
signatures. It carries one weight vector **per dimension**, because IBU unfolds
each observable separately and its weights genuinely differ between them;
`from_shared` repeats a single vector across the dimensions, which is what
OmniFold needs --- it reweights events, not observables. OmniFold draws crimson
dash-dot with triangles against IBU's green dotted squares, distinguished by
linestyle as well as colour so the panels survive greyscale printing.

The report's tables carry the third arm too: `render` reads
`metrics_omnifold.json` when it exists, and fills the two OmniFold columns with
dashes when it does not, because the template fixes the column count. Columns
run Sim, IBU, OmniFold, RAN --- the method under test last, where the eye lands,
behind what it is being compared against.

Eight columns do not fit at the default column padding. The six tables overran
the text block by ~24pt, which `pdflatex` reports as an overfull hbox and
*still compiles* --- so nothing failed and the numbers simply ran off the page.
They now set `\tabcolsep` to 4pt and drop the method name from each improvement
heading (`impr. (\%)`, unambiguous because it sits beside its method's column).
`_TABLE_COLUMNS` is the one place the count lives, and a test asserts it against
the template's own `tabular` specification --- the two have no other connection,
and had already drifted once.

## Uncertainty

`src/ran/uncertainty/` measures the variance budget: a `B x S` grid of
bootstrap datasets crossed with initialization seeds, one cell per invocation.

```bash
ran uncertainty run --cell 0 --design-dir runs/unc_x -B 8 -S 8
ran uncertainty collect --design-dir runs/unc_x -B 8 -S 8
bash scripts/submit_uncertainty.zsh                       # packed 8x8 on SLURM
```

Three things are decided there rather than left to the caller, and the package
README argues each at length.

**Three sources, not two.** Finite sample (bootstrap), split and batch order
(`data_seed`), and initialization (`seed`). Only the first is a statistical
uncertainty a measurement is obliged to report; the other two are method
variance, removable by ensembling, and quoting them inflates the band with
something a competitor eliminates by averaging. Varying `data_seed` does not
estimate the first --- every run still sees the same events --- so the design
holds it **fixed** and resamples events instead.

**A grid, not two sweeps.** Varying one axis at a fixed value of the other
gives `sigma_b^2 + sigma_eps^2` and `sigma_a^2 + sigma_eps^2`; adding those in
quadrature counts the interaction twice and overstates the total by exactly
`sigma_eps^2`. `variance.decompose` reads the full grid and returns all three
components from the balanced two-way crossed random-effects ANOVA, alongside
`naive_quadrature` so the size of the double-count is visible. Components are
moment estimators and are reported **raw**, negatives included: clamping turns
"unresolved" into "exactly zero".

**One fixed common evaluation set.** Bootstrap replicates hold different and
duplicated events, so their weight vectors are not otherwise stackable.
`reserve_evaluation_set` holds out gen-level MC events before any resampling,
which also means the evaluation sample's own finite size shifts every cell
together and cancels out of every component. It is regenerated by `collect`
from two recorded seeds rather than stored per cell.

The bin-to-bin covariance is the other output, and the one the speed result
pays for --- ~100 retrainings is not an analysis anyone runs at OmniFold's
cost. Two corrections keep it honest: the between-dataset covariance still
carries `Cov_eps / S` and is corrected for it, and RAN's weights preserve the
total count, so closure alone forces `-1 / (K - 1)` on every off-diagonal of
an equal-occupancy binning. `multinomial_off_diagonal` writes that floor next
to the measurement. `B` must also exceed the bin count: a covariance from `B`
replicates has rank `B - 1`, so at `B <= K` every correlation saturates at
`+-1` and looks like a strong result. `collect` warns rather than letting the
heatmap speak for itself.

The finalized numbers --- an 8x8 decomposition and a 100x2 covariance on the
shipped jet configuration --- are recorded in
`src/ran/uncertainty/README.md`, along with the two caveats that bound them.

## Timing

`RAN_TIMING=1` makes a run report where its wall clock went; unset, the layer
is a genuine no-op --- `phase()` hands back one shared do-nothing context
manager, so a boundary costs no `perf_counter` call and no allocation. That
matters because the timers sit inside `workflow.run` and `train.train`, which a
sweep crosses a few hundred times.

```bash
RAN_TIMING=1 ran train --dataset jets -e 100
```

Output is a Rich table on stderr plus `artifacts/timings.json` in the run
directory, written from a `finally` so a run that fell over still reports ---
the phase that raised is recorded, marked `failed`, with the time it burned
before it did.

**Phases merge by name across passes, and each carries a `pass` field.**
`scripts/submit.zsh` invokes the package three times over one run directory,
and each write used to truncate the file: the final `ran evaluate` pass left a
`timings.json` holding `evaluate` alone, with the training block --- the only
part anyone wants --- gone. A pass now replaces its own same-named phases and
leaves the rest, so the file accumulates `train`/`load`/`plots`/`evaluate`
together and `pass` says which invocation produced each row. `--run-dir` and `--load-run` name a directory up front so a crash there
still gets a file; a fresh run under the default timestamp has nowhere to write
until `_save_run` exists, and the table on stderr is then all there is.

The phases, nested ones indented under their parent:

| Phase | Covers |
| --- | --- |
| `data` | Building or loading the splits. The `Detail` column says which branch it took --- `cache hit`, `generated`, `downloaded from Zenodo` --- filled in by the loaders, which know, via `timing.note(..., to="data")` |
| `train` | The whole of `train()` |
| ` transfer` | `DeviceSplits.from_splits`, the one host->device copy of a run |
| ` compile` | XLA compiling the fused whole-run program |
| ` epochs` | Executing it |
| ` select` | `_select_by_mmd`, host-side, after the loop |
| `particle_mmd` | The particle-level diagnostic curve in `_finish_run` |
| `save` | `_save_run`: two `.keras` files plus `EpochParams` |
| `load` | `_load_artifacts`, on the `--load-run` path instead of `train`/`save` |
| `plots` | `_draw_figures`; near-zero under `--no-plots` |
| `evaluate` | `evaluate_run` |

`timings.json` is flat, with a `depth` field rather than nested objects, so a
sweep can join it against `config.json` without walking a tree. `total_seconds`
sums the **top-level** phases only: a nested one is already inside its parent
and adding it double-counts.

Three things to know before acting on a number.

**`compile` is meaningless without knowing whether the cache was warm.** A warm
XLA persistent cache (see Caching) turns 4.6s into a fraction of a second, and
that is the common case --- so a `compile` row read in isolation says compile is
free and points optimization at the wrong component. `timings.json` records
`compile_cache_warm`, sampled before the run could fill the cache.

**Timing changes how the fused path is compiled, not what it computes.**
Exposing the compile boundary needs `lower().compile()` and a call to the
compiled object, because an ordinary `jax.jit` call does both at once and shows
no seam between them. It is the same executable and the same persistent cache;
`tests/test_timing.py::TestTrainIntegration::test_timed_run_matches_an_untimed_one`
is what says so, asserting a timed run's history is bit-identical to an untimed
one's. The split is gated on `RAN_TIMING`, so the default path stays the single
call `TestFusion` pins.

**JAX is async, so phase boundaries block.** A timer stopped before the arrays
are ready charges this phase's time to whichever phase runs next, so `transfer`
and `epochs` call `jax.block_until_ready` inside the clock. This shifts when the
wait happens, never what is computed --- but a timed run and an untimed one are
not the same schedule.

## Reporting

`ran report <run_dir>` compiles one run directory into a PDF dossier at
`<run_dir>/report.pdf` --- configuration, timings, the metric tables, and
every figure, in one document. `--force` rebuilds over an existing PDF;
`--no-compile` stops at `artifacts/report.tex` so the LaTeX can be inspected
without a TeX installation.

**The run directory has two files at the top and everything else under
`artifacts/`.** `config.json` and `report.pdf` are what a human opens; the
weights, histories, metric JSON and component figures are inputs to the
report, and burying them is what made the directory readable at a glance.
`artifacts_dir(run_dir)` is the single accessor and it creates the directory,
so anything that only *reads* must use `run_dir / ARTIFACTS_DIR` instead ---
rendering a report must not mkdir into a directory it was handed.

`src/ran/templates/report.tex` is the document; `src/ran/report.py` only fills
in `<<TOKEN>>` slots and never decides layout. All rounding policy lives in
the template's `siunitx` column types, so changing how a number reads is a
LaTeX edit, not a Python one.

Two things about the figures are load-bearing:

**Jet observables are presented in physics order, not storage order.**
`JET_DISPLAY_ORDER` and `JET_VARIABLE_GROUPS` in `rantypes/constants.py` give
the twelve observables as four physics groups (mass and hard scale;
continuous angularities; splitting and 2-prong substructure; hadronization,
multiplicity and fragmentation), and `display_order()` maps a run's variables
onto it. This is presentation only --- `SUBSTRUCTURE_VARIABLES` remains the
canonical storage order and the cache key, and must not be reordered (see Jet
Column Order).

**The level figures are paginated, and the report has to agree.**
`figure_pages(dim)` is the page count for both, six panels to a page;
`report.py` emits that many `\includegraphics[page=k]` blocks without opening
the file. A run whose figures were drawn before pagination has a one-page PDF
and `pdflatex` fails with "required page does not exist" --- redraw with
`ran train --load-run <run_dir>` first. `submit.zsh` keeps them in step.

The figure pages are landscape with their own `\newgeometry{margin=8mm}`,
and two independent knobs set how they read. A panel's width on the page is
`linewidth / PANEL_COLUMNS` whatever the figure's inch size, because
`\includegraphics[width=\linewidth]` scales the figure by exactly as much as
widening it grew the figure. What the inches DO set is the rendered text size,
`font.size * linewidth_pt / (72 * figure_width_in)`. So `PANEL_COLUMNS` sizes
the panels and `PANEL_WIDTH_INCHES` sizes their labels, downwards.

Figure defects do not fail tests. Clipped labels, missing titles, overlapping
text, an occluded inset and a wrong panel aspect all passed a green suite here
and were caught only by rendering the PDF and measuring artist bounding boxes.
Render and look before claiming a plotting change works.

## Caching

Everything RAN can regenerate lives under one root, `.cache/` by default:
generated Gaussian datasets, the per-variable jet `.npz` files pulled from
Zenodo, and the XLA compilation cache. **`RAN_CACHE_DIR` moves the whole tree**,
which is what a cluster needs — on Perlmutter `$HOME` is small, quota'd and
shared across nodes:

```bash
export RAN_CACHE_DIR="$SCRATCH/ran-cache"
```

It is deliberately its own variable rather than a read of `XDG_CACHE_HOME`. That
one is already set (or defaults to `~/.cache`) on most Linux systems, so
deriving from it would silently move every existing checkout's cache and orphan
the jet data already on disk. `~` is expanded, and an empty value falls back to
the default rather than meaning the current directory — a SLURM `--export` that
forwards an unset variable delivers `""`, not absence.

`CACHE_ENV_VAR` and `CACHE_DIR` are resolved once, at import of
`rantypes/constants.py`, because the `cache_dir=` defaults throughout
`ran.data` bind to `CACHE_DIR` at import either way.

### Compilation cache

`train()` calls `_use_compilation_cache()`, which points XLA's persistent cache
at `CACHE_DIR / "jax"`. This is worth doing because **compile is the largest
single term in a short run**: `benchmarks/boundary.py` on an A100 measures 4.60s
of XLA against 0.034s per epoch, so a 100-epoch run spends half its wall clock
compiling. The cache keys on lowered HLO rather than Python identity — the fresh
`jax.jit(lambda ...)` in `_run` hits it regardless — and it lives on disk, which
is where it pays: an ensemble is N separate interpreters compiling one
architecture N times over. Measured locally, 1.41s cold → 0.36s warm across
processes.

Two settings, not one. JAX's default `jax_persistent_cache_min_compile_time_secs`
of 1.0s leaves RAN's cache **entirely empty** and says nothing about it: a run
compiles a few dozen executables totalling ~4.6s and no single one of them
clears a second. `_use_compilation_cache` drops it to zero.

Anything the caller configured wins: `JAX_COMPILATION_CACHE_DIR` and
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` still override, and an unwritable
directory costs a warning from JAX rather than the run.

## Gaussian Config Format

YAML files in `params/` use keys: `mu_gen`, `mu_true`, `sigma_gen`, `sigma_true`, `sigma_detector`. Sigma values are promoted via `sigma_to_covariance`: scalar → σ²I, vector → diag(σ²), matrix → used as-is.

## Tech Stack

- Python >= 3.14, managed with `uv` (no pip). Not 3.13: `src/ran/timing.py`
  uses PEP 758's unparenthesized `except OSError, ValueError:`, which is a
  `SyntaxError` on anything earlier -- and ruff's formatter canonicalises the
  parenthesized form to it, so it will come back if someone "fixes" it
- Keras 3 on the **JAX** backend for training; `jax[cuda13]` on x86_64 Linux
- Typer for the CLI, Rich for logging and metrics tables
- Matplotlib for publication-quality plots; `pdflatex` (TeX Live, with
  siunitx, booktabs and pdflscape) for `ran report`, and only for that
- scipy for evaluation metrics (Wasserstein distance, Jensen-Shannon divergence)
- jaxtyping + beartype for shape/dtype checking on the training loop's array seams
- ruff (lint + format), pyrefly (types, `--min-severity info`), complexipy (max 10)
- TensorFlow is not a dependency, direct or transitive. JAX is the only array
  backend in the build, so nothing here has to negotiate for the GPU.

## Backend

JAX is the only backend in the build. There is no second framework competing for
the Keras backend slot or for the GPU, so backend handling here is a one-line
default rather than a negotiation.

`src/ran/__init__.py` sets `KERAS_BACKEND=jax` and `JAX_ENABLE_X64=0`. Keras 3
still defaults to TensorFlow when the variable is unset, and TensorFlow is not
installed, so the pin is what makes `import keras` work at all — it is not
racing anything. It must still land before the first keras import, which is why
it lives in the package `__init__`; `src/ran/train.py` keeps a cheap guard that
raises a readable error if someone sets `KERAS_BACKEND` to something else by
hand.

## Precision

**RAN is float32 end to end, and the pin lives in one place:**
`EVENT_DTYPE` in `src/ran/rantypes/constants.py`, with its annotation-space
twin `EventArray` in `rantypes/types.py`. `JAX_ENABLE_X64=0` and the `dtype=`
arguments in `src/ran/models.py` follow from it. There is no dtype parameter
anywhere in the pipeline and no `astype` on the containers; there used to be
generics (`Events[T]`, `Populations[T]`, …) whose only purpose was letting IBU
carry float32 through a float64 pipeline, and with one dtype they were
ceremony.

The evidence, because this is the kind of decision that gets re-litigated:

- Every jet observable is float32-clean. `mass` and `mult` are bit-exact
  through a float32 round trip; `w`, `tau21`, `zg` and `sdm` lose exactly half
  a ULP, the minimum a cast can cost. There is no structure below float32.
- Weight normalization is stable in float32 out to a 10²⁴ weight dynamic range
  (error ~1e-8), and the batched scan reduction lands within 5e-4 of
  `min_delta`, because summing 8192-element batches then 61 partials is
  effectively pairwise summation.
- 20 paired seeds put float32 and float64 within ±0.5 percentage points of
  unfolding improvement (TOST p=0.015; paired t-test p=0.16, so no detectable
  difference). See `benchmarks/precision.py` and `benchmarks/compare_precision.py`.

Two things the pin does **not** cover:

- **It is an annotation-level contract, not a runtime one.** Nothing coerces at
  the `Populations` boundary; the checkers enforce it at author time, and the
  three data sources (`_draw_gaussian`, `load_jet_dataset`, and the
  sample-construction in `leakage.py`) narrow explicitly.
- **`ran.data.download` stays float64 on purpose.** `_get_var` upcasts before
  computing observables, because the ε it uses to protect degenerate jets is
  below the smallest float32 denormal — narrowing there would hand back `NaN`
  for exactly the jets the ε exists to protect. The narrowing happens after, in
  `load_jet_dataset`.

Five gotchas worth knowing:

- **Scores are not pinned.** Wasserstein, JS and the triangular discriminator
  are float64 and stay there. What is pinned is the data, not the measurement
  taken of it. Since the metrics moved to device (`ran.evaluate`), the
  reductions over the full sample --- the sort-and-scan behind Wasserstein, the
  histogram scatter behind the other two --- are necessarily float32, so each is
  arranged so its error is relative to the answer rather than to the largest
  intermediate: the Wasserstein scan accumulates the *signed* weights, whose
  running total is the CDF gap being measured, instead of two CDFs that both
  climb to 1 and then cancel; the histogram scatters *centered* weights and adds
  the mean back through an exact count. Everything downstream of those --- the
  divergences themselves, which are reductions over `dim x n_bins` values and so
  cost nothing --- is float64 on the host. Measured against a float64 reference
  this lands JS within 9e-9 on a CPU and ~1.3e-8 on an A100, where the
  `np.histogram` path it replaced was 5.9e-7 off. **The gap is
  platform-dependent, so do not pin a measured constant as a tolerance.** The
  scatter accumulates in float32 and the order is the hardware's choice; the
  same assertion that holds at 1.26e-8 locally returned 1.288e-8 on the
  cluster. Bound these against what `metrics.json` prints --- a tenth of the
  last printed digit --- not against the last measurement, which is what
  `TestFloat32Histograms` now does. The number is also a statement about
  *bias*, and on a GPU it is smaller than the run-to-run noise --- see the next
  bullet.
- **`metrics.json` is reproducible to ~4e-8 on a GPU, not to the last digit.**
  `_counts` bins with `empty.at[index].add(...)`, which lowers to a scatter-add;
  many events land in one bin, so on a GPU that is an *atomic* accumulation and
  the summation order is whatever the hardware chose that pass. Two
  `ran evaluate` runs over the same run directory therefore return histogram
  counts differing in the last float32 ulp, and JS values differing by ~4e-8
  relative --- measured, not estimated, and non-systematic: it moves up on some
  dimensions and down on others. On a CPU the scatter is sequential and the
  numbers repeat exactly, which is why this only ever appears on the cluster.
  It is a deliberate trade: the alternative is a sorted segment-sum or a
  one-hot matmul over the full sample for a reduction that is otherwise free.
  Two consequences. `metrics.json` prints six decimals and the sixth is not
  stable on a GPU, so a diff of two evaluations of the same run is expected to
  be non-empty; compare with a tolerance rather than by equality. And a test
  must never build the same histogram twice and compare the halves at a tight
  tolerance --- that is a determinism assertion wearing a divergence's clothes,
  and it is what
  `tests/test_evaluate_metrics.py::TestDivergencesPerDim::test_js_matches_scipy_on_a_continuous_sample`
  did until it started failing on the A100 and passing locally. Build the
  histograms once, hand the same pair to both sides. Where that is impossible
  because the double binning *is* the claim --- `TestFusedMetrics` asks whether
  the fused and unfused paths agree, and sharing a histogram would delete the
  question --- widen the tolerance instead and say why: those compare at
  `rtol=1e-6`, since the noise has been measured at 1.05e-7 relative and
  `assert_allclose`'s default `rtol` is 1e-7, which put them right on the line
  (a coin flip on the cluster, a certainty on a CPU). If bitwise reproducibility
  is ever actually needed, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` buys it
  at a throughput cost (the same flag Seeding mentions).
- **`np.float32` is not JSON-serializable.** `np.float64` subclasses Python
  `float`, so `json` accepted it silently while the pipeline was float64;
  `np.float32` raises. Anything writing numbers to JSON has to coerce first —
  see `evaluate._metric_entry`, which puts every value through `float()` on the
  way into `metrics.json` for exactly this reason.
- **`keras.ops.mean` is not float64-safe.** For float64 input it selects a
  float32 compute dtype internally and returns a float64 result carrying ~1e-8
  relative error. `src/ran/train.py` has since moved to plain `jnp`, so it is no
  longer exposed — but it still reduces with `jnp.sum(...) / n` rather than a
  mean, and `tests/test_train.py` guards the accuracy either way. Anything that
  reaches for `keras.ops` again needs to know. `ops.sum` is unaffected.
- **JAX preallocates ~75% of GPU memory on its first device allocation.** With
  TensorFlow gone there is nothing on the card to collide with, so nothing in
  the package pins itself to CPU any more — `_draw_gaussian` used to, and no
  longer does. It still matters on a shared node: `scripts/submit_uncertainty.zsh`
  gives each cell exactly one visible GPU via `srun --gpus-per-task=1`, as does
  `submit_hparam.zsh`, or the first step to start would swallow the whole card.

## Jet Column Order

For `--dataset jets`, the list of observables is an **ordering**, carried as a
`tuple[str, ...]` and never as a set. `load_jet_dataset` fills column `i` from
`variables[i]`; `_save_run` records that order in `config.json`; and
`ran evaluate` and the baselines read the recorded list back **as a list**,
in order.

This was a `frozenset`, and it produced silently wrong physics. A frozenset's
iteration order depends on the per-process randomized hashes of the strings in
it, so `ran train` built its columns in one order and recorded it, then each
later process rebuilt the same dataset in a *different* order and labelled it
with the recorded one. Every jet metric came back under the wrong observable
name, and — because the generator was trained on one column order and evaluated
against another — the reload and `ran evaluate` passes fed it permuted features
and reported large negative improvements. See `tests/test_jets.py`.

Two rules follow, both enforced rather than documented:
`load_jet_dataset` raises `TypeError` on a set (also on duplicates and unknown
names), and `cli._canonical_variables` sorts `--var` into
`SUBSTRUCTURE_VARIABLES` order, so `--var w --var m` and `--var m --var w`
describe the same run — same columns, same cache key, same `config.json`.

## Seeding

Two independent randomness axes, deliberately kept separate:

| Seed        | Set by                            | Controls                                               |
| ----------- | --------------------------------- | ------------------------------------------------------ |
| `data_seed` | `RANDataset` / `load_jet_dataset` | generation, shuffle, train/val/test split, per-epoch batch order |
| `seed`      | `train`                           | weight initialization only                             |

`train(seed=None)` draws one from system entropy and **returns the value used**,
so a run stays reproducible after the fact without deciding up front that it is
worth reproducing. Both seeds are recorded in `config.json`; configs predating
this default to `data_seed=42`, which is what those runs actually used.

The HEP ensemble — rerun on the same inputs with fresh initializations and take
the variance as the model uncertainty — is a loop over `--seed` at fixed
`--data_seed`. Because the networks are Dense-only (no dropout or batch norm)
and Adam is deterministic, the two seeds together fully determine a run, up to
non-deterministic GPU reductions. Force those with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` if bitwise reproducibility is ever
needed; it costs throughput and is not needed for variance estimates.

Batch order comes from `jax.random`, inside the trace. `train` seeds a key from
`data_seed` (carried on the splits and read by `DeviceSplits.from_splits`) and
splits it once per epoch; `train_indices(key, ...)` is a pure function of that
key, so nothing can advance the sequence out from under a caller. A second
`train` over the same `DatasetSplits` therefore sees identical data with no
rewind step — which is what the old `ArrayDataset.reset()` existed to guarantee.

`train_indices` also decides what an epoch skips. It permutes, then reshapes
into `(groups, n_disc_steps, batch_size)` — the generator updates once per
group, on the group's first batch, which is what the host loop wrote as
`step % n_disc_steps == 0`. Whatever does not fill a whole group is dropped, and
since the permutation is redrawn each epoch it is a different random tail every
pass. A split too small for one group is not an error: `n_disc_steps` clamps to
the batches available, matching what the host loop did when the rule fired only
at step 0.

`ran leakage-check` (in `src/ran/leakage.py`) depends on this: both arms must
share `--seed` or initialization variance swamps the effect and the arms differ
even with no leakage. With it fixed, detector-level results are bit-identical
between the clean and poisoned arms.

`--poison` overwrites `z_true` with `--sentinel`, defaulting to
`POISON_SENTINEL` (-999). Any far-off-manifold value works, but it must not be
`TRUTH_SENTINEL`: a truth column set entirely to that is precisely what
`Populations.create` writes when there is no truth, so `has_truth` would call
the poisoned arm truthless and `require_truth()` would refuse the particle-level
comparison the check exists to make. `run_leakage_check` rejects that value up
front rather than after a full training run.

## Training Loop

`src/ran/train.py` is hand-rolled, since the two-optimizer min-max game does not
fit `Model.fit` — but it is not a Python loop over batches. **A whole run
compiles to one XLA program.** Model state lives in JAX pytrees (`TrainState`)
for the duration, updates go through `stateless_call`/`stateless_apply`, and the
values are written back into the Keras models at the end, so the returned
objects are ordinary saveable `keras.Model`s.

The nesting, innermost out:

1. `lax.scan` over the `n_disc_steps` discriminator batches of one group.
2. One generator update per group, on the group's first batch.
3. `lax.scan` over the groups — one epoch. Then a `lax.scan` over the
   pre-batched val split, accumulating `(masked total, count)` and dividing
   once, so validation is a true mean rather than a mean of per-batch means.
4. `lax.scan` over the epochs, carrying `RunCarry` — just the state and the
   PRNG key — and emitting every epoch's `(train_d, train_g, val_d)` row
   plus its full `EpochParams`: both networks' trainable and non-trainable
   variables, stacked on a leading epoch axis. A fixed trip count is what
   lets `scan` stack that output at all; a `while_loop` cannot emit
   per-epoch arrays without a preallocated buffer of its own.

The history has three columns, not four: every one is the weighted BCE on the
same scale (`_make_pass` negates `g_loss` back before recording it), and
validation measures that BCE in exactly one place, so both networks are scored
by a single number. A `val_g` column could only repeat `val_d` — which is what
it used to hold, and what drew two identical curves on `losses.pdf`. Runs saved
before this carry the extra key; nothing reads it, and `val_d` kept its name, so
they still reload.

Nothing about model quality is decided inside the trace. Per-epoch logging
goes through `jax.debug.callback(..., ordered=True)` so the Rich handler
still sees it from inside the loop, and that is the loop's only side effect;
selection is a host-side read of what `scan` already emitted, once the run
is over.

Selection does not happen in the loop. A GAN's loss is not a proxy for a
single scalar objective monotonically related to model quality: it oscillates
around its equilibrium by construction, a flat curve cannot be told from a
stalled one, and `log 2 - BCE` estimates a divergence only when `d` is
optimal -- which nothing reports. Both criteria once built on it were unsound
and are gone.

Instead the scan emits every epoch's parameters (`EpochParams`, ~27 MB for
100 epochs of both networks), and `train` picks the epoch minimizing a
weighted MMD against a fixed subsample of the validation split. MMD is a
divergence -- zero iff the distributions match, monotone in mismatch, no
adversary and no optimization -- so patience and early stopping would be
sound again. They are gone anyway: `scan` has a fixed trip count, and at
0.034s/epoch against a 4.6s compile, early stopping saved less wall clock
than compiling the loop that implemented it.

Selection is **detector level** (`x_sim` reweighted vs `x_data`), so it needs
no truth and the method stays deployable. The particle-level MMD is computed
too, but on the host in `workflow`, never in the trace -- which is what keeps
`z_true` out of the traced program while still producing the curve. The
number reported for the restored checkpoint comes from a *test* subsample,
not the val one selection minimized.

The estimator has a resolution floor around 5e-4 in MMD^2 at m=8192, scaling
as ~1/m; below it the ranking inverts, because the empirical MMD is minimized
by weights matching the sample rather than the distribution. `benchmarks/`
measures it. `MMD_SUBSAMPLE` is 16384.

Loss math is plain `jnp` — `keras.ops` bought backend-agnosticism this module no
longer has, since `lax.scan` and `jax.random` are both native.
`stateless_call`/`stateless_apply` are the only Keras calls inside the trace.

**`train(fused=False)` is the debugging path.** It runs the identical `_epoch`
function from an ordinary Python `while` — still one XLA program per epoch, but
with breakpoints, readable tracebacks and host-side control flow. It is also the
reference the fused path is tested against (`tests/test_train.py::TestFusion`),
so the two must never diverge into separate implementations.
