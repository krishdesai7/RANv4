"""Fixed values: the Zenodo jet dataset, its cache layout, and plot metadata."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from typing import Final


CACHE_DIR: Final[Path] = Path(".cache")

ZENODO_RECORD: Final[int] = 3548091
GENERATORS: Final[tuple[str, str]] = ("Pythia26", "Herwig")
N_FILES: Final[int] = 17
SUBSTRUCTURE_VARIABLES: Final[frozenset[str]] = frozenset(
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


JET_OBS: Final[dict[str, JetVarInfo]] = {
    "m": JetVarInfo((0, 75), "Jet Mass", r"$m$ [GeV]"),
    "M": JetVarInfo((0, 80), "Jet Constituent Multiplicity", r"$M$"),
    "w": JetVarInfo((0, 0.6), "Jet Width", r"$w$"),
    "tau21": JetVarInfo(
        (0, 1.2), r"$N$-subjettiness Ratio", r"$\tau_{21}^{(\beta=1)}$"
    ),
    "zg": JetVarInfo((0, 0.5), "Groomed Jet Momentum Fraction", r"$z_g$"),
    "sdm": JetVarInfo((-14, -2), "Soft Drop Jet Mass", r"$\ln\rho$"),
}

DEFAULT_PURITY_THRESHOLD: Final[float] = np.sqrt(0.5)
