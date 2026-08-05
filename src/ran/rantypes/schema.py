from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import keras
import numpy as np

if TYPE_CHECKING:
    from typing import Any, Final, Literal

    from numpy.typing import NDArray

    from ..data import ArrayDataset
    from .types import MetricRecord, Variables

# --------------------------------
# Training
# ---------------------------------

EPS: Final[float] = keras.config.epsilon()


class TrainResult(NamedTuple):
    """What `train` returns. Unpacks as ``(g, d, history, seed)``."""

    g: keras.Model
    d: keras.Model
    history: dict[str, list[float]]
    seed: int


class TrainState(NamedTuple):
    """All mutable training state, as a JAX pytree.

    Held outside the `keras.Model`s so jitted steps stay pure and no
    host/device sync happens between steps.
    """

    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables
    opt_g: Variables
    opt_d: Variables


# ---------------------------------
# Datasets
# ---------------------------------


class GaussianConfig(NamedTuple):
    dim: int
    mu_gen: NDArray[np.double]
    mu_true: NDArray[np.double]
    cov_gen: NDArray[np.double]
    cov_true: NDArray[np.double]
    cov_detector: NDArray[np.double]


class DatasetSplits(NamedTuple):
    train: ArrayDataset
    val: ArrayDataset
    test: ArrayDataset


REQUIRED_KEYS: Final[set[str]] = {
    "mu_gen",
    "mu_true",
    "sigma_gen",
    "sigma_true",
    "sigma_detector",
}

ZENODO_RECORD: Final[int] = 3548091
GENERATORS: Final[tuple[str, str]] = ("Pythia26", "Herwig")
N_FILES: Final[int] = 17
SUBSTRUCTURE_VARIABLES: Final[set[str]] = {"m", "M", "w", "tau21", "zg", "sdm"}

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

CACHE_DIR: Final[Path] = Path(".cache")


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

# ---------------------------------
# Baselines
# ---------------------------------


@dataclass(frozen=True)
class RunConfig:
    """A validated view of a run's config.json.

    `source` is the raw dict, kept because `_load_splits` reconstructs the
    dataset from it and must see exactly what the run recorded.
    """

    source: dict[str, Any]
    dataset: Literal["gaussian", "jets"]
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]


@dataclass(frozen=True)
class UnfoldingPopulations:
    """The seven event populations every baseline needs.

    The `response_*` and `observed_reco` arrays span all three splits, since the
    response matrix and the OmniFold training set are built from every event.
    The `test_*` arrays are the test split alone, which is what gets scored.
    """

    response_gen: NDArray[np.double]
    response_sim: NDArray[np.double]
    observed_reco: NDArray[np.double]
    test_data_gen: NDArray[np.double]
    test_data_reco: NDArray[np.double]
    test_mc_gen: NDArray[np.double]
    test_mc_reco: NDArray[np.double]


@dataclass(frozen=True)
class VariableOutcome:
    variable_name: str
    status: Literal["completed", "skipped"]
    n_bins: int
    skip_reason: str | None = None


@dataclass(frozen=True)
class IBUResult:
    metrics: dict[str, MetricRecord]
    variable_names: tuple[str, ...]
    weights: NDArray[np.double]
    outcomes: tuple[VariableOutcome, ...]


DEFAULT_PURITY_THRESHOLD: Final[np.double] = np.sqrt(0.5, dtype=np.double)


# ---------------------------------
# CLI
# ---------------------------------
class LogLevel(StrEnum):
    debug = auto()
    info = auto()
    warning = auto()
    error = auto()
    critical = auto()


class DatasetName(StrEnum):
    gaussian = auto()
    jets = auto()
