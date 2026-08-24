"""pytest imports conftest before collecting anything, so doing it here makes the
guarantee hold for every file, in any order, one file at a time or all of them.
"""

import ran  # ruff: ignore[unused-import]  -- imported for its backend bootstrap
