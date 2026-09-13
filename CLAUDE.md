# RANv4

RANv4 is a Reweighting Algorithm Network that uses adversarial learning to unfold (deconvolve) reco-level data to nominal truth. The package lives under
`src/ran/`; see [docs/claude/structure.md](docs/claude/structure.md) for the full tree.

## Core Algorithm

- `(z_gen, x_sim)` are naturally paired MC events. `(z_true, x_data)` are naturally paired data events.
- **Generator** `g(z)` takes nominal-level events as input and produces per-event weights `w = g(z)`. For data events (y=1), weights are fixed to 1. Weights are normalized to preserve total counts per class.
- These weights are applied to reco-level _distributions_ (not the events themselves). `x_data` is reweighted with w=1 (unchanged), `x_sim` is reweighted with `w = g(z_gen)`.
- **Discriminator** `d(x)` operates at reco level to distinguish these two reweighted distributions.
- **Loss**: weighted BCE — `w_i * y_i * log(d(x_i)) + (1 - y_i) * w_i * log(1 - d(x_i))`.
- **Training**: min-max game. `d` minimizes BCE (correctly distinguish data from reweighted sim). `g` maximizes BCE (generate weights that confound `d`).

## Critical Constraint

**No network should ever have access to `z_true`.** This is the unfolded truth that we do not know in principle. `g` only sees `z` (nominal-level features), never the true particle-level values from data. The event types in [docs/claude/data-model.md](docs/claude/data-model.md) enforce this in the type system, not just by convention.

## Tooling Preferences

- Prefer `fd` over `find`, `rg` over `grep`, and `fzf` for fuzzy finding. `find`/`grep` are fine as fallbacks.
- When working with Python, you may invoke the relevant `/astral:<skill>` for uv, ty, and ruff.

## Reference Docs

Everything below is detail this file used to carry inline. Read the linked file when a task touches that area; don't load all of them speculatively.

| Doc                                                              | Covers                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [docs/claude/data-model.md](docs/claude/data-model.md)           | `Populations`/`ZXY`/device-resident split types; jet column ordering     |
| [docs/claude/structure.md](docs/claude/structure.md)             | Full source tree, `runs/` and `.cache/` layout                           |
| [docs/claude/cli-and-running.md](docs/claude/cli-and-running.md) | CLI reference, `just` recipes, test markers, `scripts/submit.zsh`        |
| [docs/claude/training-loop.md](docs/claude/training-loop.md)     | `train.py`'s fused `lax.scan` program and host-side checkpoint selection |
| [docs/claude/seeding.md](docs/claude/seeding.md)                 | `data_seed` vs `seed`, batch order, leakage-check                        |
| [docs/claude/precision.md](docs/claude/precision.md)             | The float32 pin, where it doesn't reach, and reproducibility gotchas     |
| [docs/claude/tech-stack.md](docs/claude/tech-stack.md)           | Dependencies, Keras/JAX backend pin, Gaussian config YAML format         |
| [docs/claude/caching.md](docs/claude/caching.md)                 | `RAN_CACHE_DIR`, dev-tool caches, the XLA compilation cache              |
| [docs/claude/reporting.md](docs/claude/reporting.md)             | `ran report`, the LaTeX template, figure pagination                      |
| [docs/claude/timing.md](docs/claude/timing.md)                   | `RAN_TIMING=1`, `timings.json` phase structure                           |
| [docs/claude/omnifold.md](docs/claude/omnifold.md)               | The OmniFold baseline's subprocess boundary and SLURM job                |
| [docs/claude/uncertainty.md](docs/claude/uncertainty.md)         | The bootstrap x seed variance design                                     |
| [docs/claude/releasing.md](docs/claude/releasing.md)             | The Actions release workflow and versioning policy                       |

Module-level detail also lives in `README.md` files next to the code: `src/ran/README.md`, `src/ran/rantypes/README.md`, `src/ran/data/README.md`, `src/ran/baselines/README.md`, `src/ran/uncertainty/README.md`, `benchmarks/README.md`, `scripts/README.md`.

## Maintaining This File

Keep this file to roughly 100 lines. It is the index, not the encyclopedia: new subsystem detail, war stories and rationale belong in a file under `docs/claude/` (add a new one if none fits) or a module's own `README.md`, linked from the Reference Docs table above rather than inlined here. If a change needs more than a couple of sentences here, it almost certainly belongs in one of those instead.

## Quick Start

```zsh
uv run ran train --config params/1d_default.yaml   # Gaussian
uv run ran train --dataset jets                    # all twelve jet variables
uv run ran evaluate --run-dir runs/2026-...
uv run ran report runs/2026-...
uv run just validate                               # format, lint, types, complexity, tests
uv run just test-fast                              # the same suite minus @pytest.mark.slow
```
