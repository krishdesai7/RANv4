"""CLI choice enums.

`ran.cli` imports these while it is still backend-free, which is why they live
here rather than beside the code they select for.
"""

from __future__ import annotations

from enum import StrEnum, auto


class LogLevel(StrEnum):
    debug = auto()
    info = auto()
    warning = auto()
    error = auto()
    critical = auto()


class DatasetName(StrEnum):
    gaussian = auto()
    jets = auto()
