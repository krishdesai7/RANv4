"""What the baselines take in and hand back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray

    from .events import Populations
    from .types import MetricRecord


class UnfoldingPopulations(NamedTuple):
    """What every baseline needs: a sample to unfold with, a sample to score on.

    `full` spans every split and supplies the response (`full.mc`) and the
    measurement (`full.data`). `test` is the held-out split alone, which is
    where the metrics are computed.
    """

    full: Populations
    test: Populations


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
