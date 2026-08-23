from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray

    from .events import Populations
    from .types import MetricRecord


class UnfoldingPopulations(NamedTuple):
    full: Populations
    test: Populations


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
