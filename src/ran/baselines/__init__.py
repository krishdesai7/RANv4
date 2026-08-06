"""Comparison baselines: IBU and OmniFold.

Only the backend-agnostic helpers from :mod:`._shared` are re-exported. The two
baseline modules are deliberately *not* imported here, because they need
different Keras backends and there is one backend per interpreter:
:mod:`.omnifold` hard-sets ``KERAS_BACKEND=tensorflow`` at import and must be
the entry point of its own process, while :mod:`.ibu` reaches :mod:`ran.train`,
which loads only on JAX. Re-exporting both would make each unimportable via the
other -- and would leak the losing backend into every subprocess. Import the
one you want directly::

    from ran.baselines.ibu import evaluate_runs
    from ran.baselines.omnifold import omnifold_unfold
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._shared import evaluate_dimension as evaluate_dimension
from ._shared import load_populations as load_populations
from ._shared import parse_run_config as parse_run_config
from ._shared import prepare_populations as prepare_populations

if TYPE_CHECKING:
    from typing import Final

__all__: Final[list[str]] = [
    "evaluate_dimension",
    "load_populations",
    "parse_run_config",
    "prepare_populations",
]
