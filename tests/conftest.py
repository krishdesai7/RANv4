"""Pin the Keras backend before any test module imports `keras`.

The backend is fixed at the first `keras` import, and `ran/__init__` is what
defaults it to JAX. A test module cannot be relied on to establish that itself:
imports there are isort-ordered, so `import keras` sorts above `import ran` and
runs first. That left the suite passing only because `tests/test_cli.py`
happened to be collected first and imported `ran` on the way; running
`tests/test_train.py` on its own picked up whatever backend the environment
defaulted to, and `ran.train` refused to load.

pytest imports conftest before collecting anything, so doing it here makes the
guarantee hold for every file, in any order, one file at a time or all of them.

`setdefault` semantics are preserved: this only defaults the backend, so
`tests/test_omnifold.py` -- which needs TensorFlow -- can still override it. The
import below does not pull in `keras` itself, so that override is still in time.
"""

import ran  # ruff: ignore[unused-import]  -- imported for its backend bootstrap
