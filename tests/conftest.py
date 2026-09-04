"""pytest imports conftest before collecting anything, so doing it here makes the
guarantee hold for every file, in any order, one file at a time or all of them.
"""

import os
import uuid

import pytest
import ran  # ruff: ignore[unused-import]  -- imported for its backend bootstrap
from ran.rantypes import constants


def _default_cache_is_writable() -> bool:
    """Whether the process can write RAN's default (non-`tmp_path`) cache dir.

    A few dataset/workflow tests build a `RANDataset` without an explicit
    `cache_dir`, so they generate into `constants.CACHE_DIR`. On a locked-down
    filesystem (sandboxed local runs, read-only `/.cache`) that write fails with
    `OSError`; on HPC, where these are meant to run, it succeeds. Probe once.
    """
    path = constants.CACHE_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".writable-probe-{uuid.uuid4().hex}"
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


_CACHE_WRITABLE = _default_cache_is_writable()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "writes_default_cache: needs a writable RAN default cache dir "
        "(skipped where the filesystem is read-only, e.g. local sandbox runs; "
        "force with RAN_RUN_CACHE_TESTS=1)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "writes_default_cache" not in item.keywords:
        return
    if _CACHE_WRITABLE or os.environ.get("RAN_RUN_CACHE_TESTS"):
        return
    pytest.skip(f"default cache dir {constants.CACHE_DIR} is not writable")
