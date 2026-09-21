# Configuration

`ran` resolves every layerable option through a five-layer stack, matching the
convention uv, ruff and pyrefly use, so a machine-level or project-level
preference lives in a file instead of a person's shell history or a
`RUN_ARGS` string wedged into a submit script.

| # | Layer | Location |
| --- | --- | --- |
| 1 | Code default | the Typer signature in `cli.py` |
| 2 | Global config | `$XDG_CONFIG_HOME/ran/deconvolve.toml`, default `~/.config/ran/deconvolve.toml` |
| 3 | Project config | nearest `deconvolve.toml`, or `[tool.deconvolve]` in `pyproject.toml` |
| 4 | Environment | `RAN_*`, via Typer's existing `envvar=` |
| 5 | Command line | `--n-epochs 500` |

Each layer overrides everything above it. Layers 2-3 are files `src/deconvolve/config.py`
discovers and merges into a Click `default_map`; the module is a stdlib-only
leaf (`tomllib`, `pathlib`, `difflib`) so discovery and merge stay unit-
testable without importing JAX, Keras or Typer. It does not import `os` itself
— `discover()` takes the environment mapping as a parameter, injected by the
caller (`cli.py` passes `os.environ`), which is what keeps this module
testable against an arbitrary environment rather than the process's real one.

## Discovery for layer 3

Walk upward from the current working directory. The first directory holding
either a `deconvolve.toml` or a `pyproject.toml` with a `[tool.deconvolve]` table wins
outright, and the walk stops there — within that directory, `deconvolve.toml`
shadows `pyproject.toml` entirely; they are never merged with each other. The
walk never passes the enclosing git repository root (`.git`, checked as a file
or a directory so worktrees and submodules bound it too); outside any
repository it terminates at the filesystem root.

```
~/work/          deconvolve.toml            ignored, walk already stopped
  repo/    .git/
    a/          pyproject.toml       [tool.deconvolve]  <- wins, walk stops
      b/        (cwd)
```

Exactly one project-level file ever contributes, so "which file did this value
come from" has exactly one answer — the alternative, merging every file on the
path, was rejected for exactly this reason (see the design spec's
"Alternatives considered"). `src/deconvolve/config.py`'s `_project_layer` implements
the walk; `_project_layer_in` implements the shadowing within one directory.

## Schema

The TOML mirrors the command tree, because Click's `default_map` is keyed by
command path:

```toml
# deconvolve.toml --- the same tables, without the `tool.` prefix
[train]
n-epochs = 500
batch-size = 2048

# pyproject.toml
[tool.deconvolve]
log-level = "debug"          # a group-level option

[tool.deconvolve.train]
n-epochs = 500
lr-g = 3e-5

[tool.deconvolve.uncertainty.freeze]
n-eval = 100000
```

`deconvolve.toml` drops the `tool.ran` prefix: a top-level table names a command, a
top-level key names a group-level option (today just `log-level`, the only
thing declared on `@app.callback()` itself). `[train]` in `deconvolve.toml` and
`[tool.deconvolve.train]` in `pyproject.toml` are the same table.

**Tables are per-command, not one flat namespace**, because the same name
means different things in different commands: `n_epochs` is 100 for `train`
and `uncertainty run`/`freeze`, but 50 for `baseline omnifold`; `batch_size`
is 1024 in most commands but 512 for `baseline omnifold`
(`cli.py:omnifold_command`). A flat `n-epochs = 500` would silently retune a
baseline while the author believed they were configuring training.
`config_spec.build_spec` derives the tree straight from the live Typer app —
command names, their option names, the denylist below — so a new command or
flag is validated the moment it is added, with no parallel table to keep in
sync.

Keys are kebab-case (`n-epochs`), matching uv and ruff; underscores are
accepted and normalized (`-` to `_` on lookup), since the Python parameter is
`n_epochs` and both spellings get typed.

**Option keys always normalize `-` to `_`; command names never do.** A table
name like `[leakage_check]` is looked up against `spec.children` literally,
with no normalized fallback — the registered command is named
`leakage-check` (with a hyphen), so the underscored spelling is an unknown
table, rejected with a `did you mean leakage-check?` suggestion rather than
silently resolved. This holds because no registered command name contains an
underscore standing in for a hyphen; if one ever did, this asymmetry would
need revisiting.

## What is not layerable

Every Typer **Option** is layerable except:

- Every positional **Argument** (`report`'s `run_dir`).
- `design_dir` and `cell` on `uncertainty run` — per-job identity. The whole
  point of a SLURM array is that these differ per invocation. But see below:
  `uncertainty run` is excluded by a different, stronger mechanism than the
  denylist that rejects these two by name on every other command.
- `design_dir`, and `n_datasets`/`n_seeds`/`data_seed`/`init_seed`, on
  `uncertainty collect` — the design's grid shape, which `design.json` is the
  single authority for once `freeze` has run (see "The freeze path"); an
  ambient file must not be able to reach these even though `collect` itself
  is an ordinary layered command. `n_bins` stays layerable: it is a
  presentation choice for the figure, not part of the grid.
- `force` on every command that has one (`evaluate`, `report`, `baseline ibu`,
  `baseline omnifold`, `uncertainty freeze`) — a destructive-rebuild toggle
  should be typed each time, not silently inherited from a file nobody is
  looking at when they run the command.
- `load_run` on `train` — names one specific prior run, not a preference.
- `as_json` on `config show` — a per-invocation output-format toggle.

`config_spec.NOT_LAYERABLE` keys this denylist by command path (empty tuple
for the root) so the error names which command rejected the key. A config key
naming one of these is a hard `ConfigError` explaining why, not merely
"unknown key".

**`uncertainty run` is excluded by a different mechanism**, not the denylist
above: `config_spec.FROZEN_COMMANDS` removes it from the command tree
`build_spec` walks entirely, so a `[uncertainty.run]` table anywhere is an
unknown-table error regardless of what it contains, and no discovered file
can populate any part of its `default_map` even by accident. `NOT_LAYERABLE`
rejects specific keys within a command that is otherwise ordinarily layered
(as `uncertainty collect` is); `FROZEN_COMMANDS` removes a whole command from
the layered stack. See "The freeze path" below for why `run` needs the
stronger of the two.

## Resolving once per invocation

The stack is resolved exactly once per invocation, in the root `configure()`
callback (`cli.py`), and the resulting `Resolved` (or, for `config show`
alone, the `ConfigError` it tolerated) is stashed on `ctx.meta` — shared by
every `Context` in one invocation's chain — for `train`, `uncertainty freeze`
and `config show` to read back rather than re-resolve. Click's own
`ctx.get_parameter_source` reflects that same resolution (it is set during
the same walk that builds `ctx.default_map`), so a second, independent
`load()` call — which earlier revisions of this feature made from each of
those three commands — risked disagreeing with it if the discovered files
changed between the two calls: a file appearing mid-invocation, an NFS
attribute-cache refresh, another node writing to a shared filesystem. That
would make `config.json`'s `_origin` block, or `design.json`'s, name a file
that did not actually supply the value it is attached to — provenance that is
wrong rather than absent, which for scientific software is the worse of the
two failure modes. Resolving once removes the possibility along with the
redundant filesystem walk.

## `deconvolve config show`

```bash
deconvolve config show            # every file-supplied setting
deconvolve config show train      # scoped to one command
deconvolve config show --json     # same content, for scripting
```

This answers **"what did the config files say"**, not "what is every option's
effective value". `_values_table` iterates `resolved.values`, which only ever
holds keys a *file* actually supplied. An option left at its code default, or
set only through a `RAN_*` environment variable, does not appear — verified:
`RAN_LOG_LEVEL=debug deconvolve config show` with no config files anywhere renders
an empty values table. This is a deliberate scope decision, not a bug to fix
by enumerating every option; for the fully resolved picture, use
`deconvolve <command> --help` (which renders the effective default Click would
apply for that command) or a run's `config.json` `_origin` block, which
records where every value that actually reached that run came from —
including code defaults and environment variables, not just file-supplied
ones.

Renders three things: which files were found and in what role (`global` /
`project`), each *file-supplied* value next to a short origin
(`deconvolve.toml (project)`), and an "Environment-only (not layered)" section for
`DECONVOLVE_CACHE_DIR`/`DECONVOLVE_TIMING` (see below) — that section is a fixed, hand-
written pair of names, unrelated to `_values_table`'s per-value origins. The
short origin disambiguates the case where the global and project files share
the exact filename `deconvolve.toml` — without the role suffix, two different files
would render identically.

`--json` emits the fuller `"<filename>:<absolute path>"` origin string instead
of the short label, because a script consuming it needs the actual path, not
a display shorthand. It also emits only file-supplied keys, same as the
table, scoped the same way by a `command` argument.

The origins shown for the keys that *do* appear are read from the same
`Resolved` object that feeds `config.json`'s `_origin` block
(`workflows/train.py`), so where the two overlap they cannot disagree. But
they answer different questions over different key sets — `config show` only
ever knows about file-supplied keys, `_origin` knows about every key that
reached the run, file-supplied or not — so it is not accurate to say they
"cannot drift apart" in general. A broken config file does not disable
`config show` itself: the root `configure()` callback in `cli.py` tolerates a
`ConfigError` only when the invoked subcommand is `config`, stashes it on
`ctx.meta`, and `config_show_command` reports it instead of the values table.

## The freeze path

A variance design (`notes/agent/uncertainty.md`) is a SLURM array of B x S
independent `deconvolve uncertainty run` invocations, all on a shared filesystem
reading the same working directory. If `uncertainty run` resolved from the
ordinary config stack, an edit to `deconvolve.toml` after cell 30 had already started
would change what cells 30-63 measure — the array would silently split into
two designs sharing one output directory, and nothing in the recorded weights
would reveal it. The bootstrap-versus-seed decomposition depends on every cell
having trained under identical settings; a silently split design produces a
number that looks like a variance estimate and is not one.

So `uncertainty run` does not read the config layers at all. `config_spec.
FROZEN_COMMANDS` excludes `("uncertainty", "run")` from the command tree
`build_spec` walks, which means no discovered file can populate its
`default_map` even by accident. Instead:

```bash
deconvolve uncertainty freeze --design-dir DIR [options]   # once, on the login node
deconvolve uncertainty run --cell N --design-dir DIR       # once per cell, in the array
```

`freeze` resolves the full five-layer stack exactly once and writes
`DIR/design.json`: the resolved values under `"config"` and their origins
under `"_origin"` (`uncertainty/design.py:freeze_design`). It opens the file
with mode `"x"` rather than checking `.exists()` first, so the refusal to
overwrite an existing `design.json` is atomic — not a check a racing process
could slip past. `--force` is accepted but still refused if any `cell_*.npz`
already exists in the directory (`freeze_design`'s `CELL_GLOB` check): a
design cannot be re-frozen once cells have started writing, which is the exact
mid-array edit this whole mechanism exists to prevent.

Each cell then calls `load_frozen`, which requires `design.json` to exist and
to carry every key `freeze` would have written — a truncated, hand-edited, or
older-`freeze` file is rejected by name at cell 0, `_require_complete` in
`cli.py`, rather than one key silently falling back to a bare code default a
different cell would not share. Absent entirely, the error names `ran
uncertainty freeze` directly. This is a breaking change from the previous
workflow, where a design directory had no configuration file at all and every
cell just used whatever flags `scripts/submit_uncertainty.zsh` happened to
pass.

Precedence inside a cell is `COMMANDLINE > design.json > code default` —
strictly narrower than the ordinary stack. `_resolve_cell_settings` checks
`ctx.get_parameter_source(name) is COMMANDLINE` per parameter and only then
lets a typed flag override the frozen value; everything else, including an
exported `RAN_*` environment variable, is ignored. Environment variables are
excluded on purpose: an exported `RAN_N_EPOCHS` in a batch script is exactly
as capable of splitting a design as an edited `deconvolve.toml` is, and a hand-rerun
of one failed cell under a different shell environment must train under the
same settings as the rest of the array.

`scripts/submit_uncertainty.zsh` calls `freeze` on the login node, between
creating the design directory and calling `sbatch`; every cell inside the job
script invokes only `--cell` and `--design-dir`.

## Deferred: `cache-dir` and `timing`

`DECONVOLVE_CACHE_DIR` and `DECONVOLVE_TIMING` stayed environment-only and did not join the
stack. Both resolve at **import** — `CACHE_DIR` at `coretypes/constants.py:35`,
the timing recorder at `instrumentation/timing.py:122` — and get bound into
default arguments (`data/jets.py:62`, `data/download.py:272`,
`data/datasets.py:153`) before the `configure()` Typer callback has run, let
alone before it could assign a `default_map`. Folding them in would mean
turning module-level constants into lazy accessors and unbinding three
default arguments in code that decides where datasets are cached — a change
with its own reproducibility risk, not a documentation-sized one. `deconvolve config
show` lists both in a separate "Environment-only" section so the boundary is
visible rather than a surprise, and `notes/agent/caching.md` records the same
reasoning next to the rest of the caching discussion.
