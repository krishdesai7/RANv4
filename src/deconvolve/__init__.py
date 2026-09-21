# pyrefly: ignore-errors[unused-call-result]
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

os.environ.setdefault(key="KERAS_BACKEND", value="jax")
os.environ.setdefault(key="JAX_ENABLE_X64", value="0")

from . import cli
from .cli import (
    app,
    configure,
    evaluate_command,
    ibu_command,
    leakage_check_command,
    train_command,
)

__all__: Final[Sequence[str]] = (
    "app",
    "cli",
    "configure",
    "evaluate_command",
    "ibu_command",
    "leakage_check_command",
    "train_command",
)
