from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from .events import Populations
    from .types import MetricRecord


class UnfoldingPopulations[T: np.floating = np.double](NamedTuple):
    full: Populations[T]
    test: Populations[T]

    def astype[U: np.floating](self, dtype: type[U]) -> UnfoldingPopulations[U]:
        return UnfoldingPopulations[U](
            full=self.full.astype(dtype), test=self.test.astype(dtype)
        )


@dataclass(frozen=True)
class VariableOutcome:
    variable_name: str
    status: Literal["completed", "skipped"]
    n_bins: int
    _: KW_ONLY
    skip_reason: str | None = None


@dataclass(frozen=True)
class IBUResult:
    metrics: dict[str, MetricRecord]
    variable_names: tuple[str, ...]
    weights: NDArray[np.single]
    outcomes: tuple[VariableOutcome, ...]
