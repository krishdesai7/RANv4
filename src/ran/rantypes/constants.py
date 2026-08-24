from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from typing import Final, LiteralString


# The one floating type the pipeline carries, end to end.
#
# The jet inputs justify it: `mass` and `mult` are bit-exact through a float32
# round trip, and the other four observables lose exactly half a ULP, so there
# is no structure below float32 to preserve. An ensemble of 20 paired seeds put
# float32 and float64 within +/-0.5 percentage points of unfolding improvement
# (TOST p=0.015); see `benchmarks/precision.py`. Everything downstream --- the
# containers, the models, `JAX_ENABLE_X64` --- follows from this line.
EVENT_DTYPE: Final[type[np.single]] = np.single

# Everything RAN can regenerate lives under one root: the dataset `.npz` caches
# and the XLA compilation cache. `RAN_CACHE_DIR` relocates the whole tree, which
# is what a cluster needs --- on Perlmutter `$HOME` is small, quota'd and shared
# across nodes, `$SCRATCH` is none of those, and hardcoding either would be
# wrong for everyone not on that machine.
#
# It is deliberately its own variable rather than a read of `XDG_CACHE_HOME`.
# That one is already set (or defaults to `~/.cache`) on most Linux systems, so
# deriving from it would silently move every existing checkout's cache the first
# time this version ran, orphaning the ~2GB of Zenodo jet data already on disk.
# A project-local `.cache/` stays the default because `.gitignore` covers it.
#
# Read once, at import: the module-level constant is what the `cache_dir=`
# defaults below bind to, and those bind at import either way.
CACHE_ENV_VAR: Final[LiteralString] = "RAN_CACHE_DIR"
CACHE_DIR: Final[Path] = Path(os.environ.get(CACHE_ENV_VAR) or ".cache").expanduser()

# XLA keys its persistent cache on lowered HLO plus the jaxlib and backend
# versions, so a stale entry is a miss rather than a wrong answer -- upgrading
# JAX or changing an architecture costs a recompile, never a wrong number.
#
# It is unbounded by default (`jax_compilation_cache_max_size` is -1), but a run
# adds only a few MB and only for a shape/config it has not seen, so it plateaus
# rather than grows. Set that config if a shared directory needs a ceiling; it
# also turns on the file lock, which is otherwise absent.
COMPILE_CACHE_DIR: Final[Path] = CACHE_DIR / "jax"

RUN_DIR: Final[Path] = Path("runs")
ZENODO_RECORD: Final[int] = 3548091
GENERATORS: Final[tuple[LiteralString, LiteralString]] = ("Pythia26", "Herwig")
N_FILES: Final[int] = 17
# A tuple, emphatically not a `frozenset`. These names select *columns*, and
# `load_jet_dataset` fills column `i` from the `i`-th name --- so the container
# holding them is an ordering, and a set has none. It used to be a frozenset,
# whose iteration order depends on the per-process randomized hashes of the
# strings inside it: `ran train` built its columns in one order and recorded
# that order in `config.json`, then `ran baseline ibu` and `ran evaluate`
# rebuilt the same dataset in a *different* order in their own processes and
# labelled it with the recorded one. Same six observables, six wrong names ---
# and worse, a generator trained on one column order evaluated against another.
# The order here matches `JET_OBS` below.
SUBSTRUCTURE_VARIABLES: Final[tuple[LiteralString, ...]] = (
    "m",
    "M",
    "w",
    "tau21",
    "zg",
    "sdm",
)

# Cache-safe filenames: avoid case collisions on case-insensitive filesystems
# (macOS APFS default), where "m.npz" and "M.npz" resolve to the same path.
CACHE_FILENAMES: Final[dict[str, str]] = {
    "m": "mass",
    "M": "mult",
    "w": "w",
    "tau21": "tau21",
    "zg": "zg",
    "sdm": "sdm",
}


class JetVarInfo(NamedTuple):
    xlim: tuple[float, float]
    xlabel: str
    symbol: str


LOG_RHO_FLOOR: Final[float] = -14.0
JET_OBS: Final[dict[str, JetVarInfo]] = {
    "m": JetVarInfo(xlim=(0, 75), xlabel="Jet Mass", symbol=r"$m$ [GeV]"),
    "M": JetVarInfo(xlim=(0, 80), xlabel="Jet Constituent Multiplicity", symbol=r"$M$"),
    "w": JetVarInfo(xlim=(0, 0.6), xlabel="Jet Width", symbol=r"$w$"),
    "tau21": JetVarInfo(
        xlim=(0, 1.2),
        xlabel=r"$N$-subjettiness Ratio",
        symbol=r"$\tau_{21}^{(\beta=1)}$",
    ),
    "zg": JetVarInfo(
        xlim=(0, 0.5), xlabel="Groomed Jet Momentum Fraction", symbol=r"$z_g$"
    ),
    "sdm": JetVarInfo(
        xlim=(LOG_RHO_FLOOR, -2), xlabel="Soft Drop Jet Mass", symbol=r"$\ln\rho$"
    ),
}

DEFAULT_PURITY_THRESHOLD: Final[np.double] = np.sqrt(0.5)
TRUTH_SENTINEL: Final[np.double] = np.double(np.iinfo(int_type=np.short).min)

# What `ran leakage-check --poison` overwrites z_true with. Any far-off-manifold
# value does the job, so this is only a default -- but it must not be
# TRUTH_SENTINEL. A truth column set entirely to that value is exactly what
# `Populations.create` writes when there is no truth at all, so `has_truth`
# would report the poisoned arm as having none and `require_truth()` would
# refuse the particle-level comparison the check exists to make.
POISON_SENTINEL: Final[np.double] = np.double(-999.0)
