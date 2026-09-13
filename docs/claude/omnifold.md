# OmniFold

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
documented — `pyproject.toml` pins `[tool.ruff.per-file-target-version]` for
`**/*_worker.py` to `py313`, pyrefly excludes the same glob, and
`tests/test_omnifold.py::TestQuarantine` asserts the module is absent from
`sys.modules` and that TensorFlow is not importable at all.

**The ruff pin matters.** Ruff infers `py314` from `requires-python`, and its
formatter rewrites `except (A, B):` into PEP 758's unparenthesized form — a
`SyntaxError` on 3.13. A test compiles the worker to catch a regression.

**`uv` must be on `PATH` at runtime**, since it is what provisions the worker.
Its absence is translated into a readable message rather than a
`FileNotFoundError` from inside `subprocess`, because the fix is an install.

**The worker runs under `PYTHONSAFEPATH=1`, and must.** A script's own
directory goes on `sys.path[0]`, and the worker's directory is
`src/ran/baselines/`, which contains `omnifold.py`. Without the flag the
worker's `from omnifold import MLP, DataLoader, MultiFold` resolves to the
host half rather than the installed package. `PYTHONSAFEPATH` stops the
interpreter from prepending that directory; the worker imports nothing local,
so it loses nothing. `TestTheWorkerDoesNotImportThisPackage` reproduces the
collision with a poisoned sibling.

**The worker environment is not in `uv.lock`.** uv resolves the PEP 723 header
on first use, which needs outbound network, and compute nodes generally have
none. Warm it on a login node, the way the jet cache is warmed:

```bash
uv run --no-project src/ran/baselines/_omnifold_worker.py
```

## It runs on the CPU, silently, without a CUDA 12 toolkit

**On Perlmutter `module load cudatoolkit/12.9` is mandatory.** The default
environment leads `LD_LIBRARY_PATH` with four CUDA **13.2** trees and the
`tensorflow` wheel is a CUDA **12** build; exactly one library goes unreachable,
`libcusolver.so.11`, and one is enough for TF to skip registering every GPU. It
then runs on the CPU and **raises nothing** — the weights come back correct,
tens of times slower, and the baseline looks like it worked.

So the worker reports the device it used and `_warn_if_on_cpu` warns when it was
not a GPU. That warning is the only signal this failure produces; do not silence
it. `benchmarks/gpu_coexistence.py` measures the whole thing and its README
section records the numbers.

What that benchmark also settled: a TensorFlow subprocess gets the GPU **even
with JAX's default 75% preallocation held by the parent**. The worker peaks at
1.07GB against the 9.4GB that survives, so no `XLA_PYTHON_CLIENT_*` tuning is
needed.

## Its own job, not a step in `submit.zsh`

`scripts/submit.zsh` does not run OmniFold. That job asks for
`--time=00:15:00`, and OmniFold alone measured **~41 minutes** on the shipped
configuration (1.6M samples, twelve observables, `niter=3`, 50 epochs), so it
would not fit in what is left after RAN trains. It gets
`scripts/submit_omnifold.zsh` instead, which takes an existing run directory and
asks for 75 minutes:

```zsh
sbatch scripts/submit_omnifold.zsh runs/<timestamp>Z
```

That script loads `cudatoolkit/12.9`, runs the baseline, unloads it, redraws the
figures, re-scores and rebuilds the report. **The module unload is an EXIT trap,
not zsh's `{ } always { }`**, because `always` does not run under `set -e`,
which ERR_EXIT leaves before reaching — a failed unfolding would otherwise
leave the CUDA 12 toolkit loaded over whatever ran next in the allocation.

**`module` is not available in a batch script until it is initialised.** It is
a shell function Lmod defines in a startup file that only an interactive or
login shell reads, so a batch script otherwise dies with
`command not found: module`. `scripts/_lmod.zsh` initialises it, and every
script calling `module` sources that first; `tests/test_scripts.py` asserts
the ordering, and `zsh -n` parses every script.

## Getting OmniFold onto the figures

Presence is the mechanism, the same one IBU uses.
`_load_baseline_weights` returns one `BaselineOverlay` per `*_weights.npz` that
exists in `artifacts/` when the figures are drawn, so:

```zsh
ran baseline omnifold --run-dir runs/<timestamp>Z   # writes omnifold_weights.npz
ran train --load-run runs/<timestamp>Z              # reloads, redraws with it
```

`--load-run` reloads the saved generator rather than training, so the redraw is
cheap and the run is untouched.

`plotting.BaselineOverlay` carries one weight vector **per dimension**, because
IBU unfolds each observable separately and its weights genuinely differ between
them; `from_shared` repeats a single vector across the dimensions for OmniFold,
which reweights events, not observables. OmniFold draws crimson dash-dot with
triangles against IBU's green dotted squares, distinguished by linestyle as
well as colour so the panels survive greyscale printing.

The report's tables carry the third arm too: `render` reads
`metrics_omnifold.json` when it exists, and fills the two OmniFold columns with
dashes when it does not, because the template fixes the column count. Columns
run Sim, IBU, OmniFold, RAN — the method under test last, where the eye lands,
behind what it is being compared against.

Eight columns need `\tabcolsep` set to 4pt and the method name dropped from
each improvement heading (`impr. (\%)`, unambiguous because it sits beside its
method's column) to fit at default padding. `_TABLE_COLUMNS` is the one place
the column count lives, and a test asserts it against the template's own
`tabular` specification.
