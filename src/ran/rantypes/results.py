"""What the baselines take in and hand back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from .events import Populations
    from .types import MetricRecord


class UnfoldingPopulations[T: np.floating = np.double](NamedTuple):
    """What every baseline needs: a sample to unfold with, a sample to score on.

    `full` spans every split and supplies the response (`full.mc`) and the
    measurement (`full.data`). `test` is the held-out split alone, which is
    where the metrics are computed.
    """

    full: Populations[T]
    test: Populations[T]

    def astype[U: np.floating](self, dtype: type[U]) -> UnfoldingPopulations[U]:
        """Both samples at another precision; see `Populations.astype`."""
        return UnfoldingPopulations[U](self.full.astype(dtype), self.test.astype(dtype))


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
    # Single precision, matching the published IBU results -- see `ibu`.
    weights: NDArray[np.single]
    outcomes: tuple[VariableOutcome, ...]
