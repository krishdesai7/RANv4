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
