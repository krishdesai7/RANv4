from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from typing import Final, LiteralString


CACHE_DIR: Final[Path] = Path(".cache")
RUN_DIR: Final[Path] = Path("runs")
ZENODO_RECORD: Final[int] = 3548091
GENERATORS: Final[tuple[LiteralString, LiteralString]] = ("Pythia26", "Herwig")
N_FILES: Final[int] = 17
SUBSTRUCTURE_VARIABLES: Final[frozenset[LiteralString]] = frozenset(
    ("m", "M", "w", "tau21", "zg", "sdm")
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
