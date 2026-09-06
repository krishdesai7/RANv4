"""Optional wall-clock instrumentation for a run's phases.

Off unless `RAN_TIMING` is set, and *off* means a shared no-op context manager:
no `perf_counter`, no allocation, nothing appended. That matters because the
timers sit at phase boundaries inside `workflow.run` and `train.train`, which a
sweep crosses a few hundred times.

The point of the layer is to say which component to go optimize, so the report
is shares of the total rather than raw seconds alone, and nested phases are
recorded with their depth so `train.compile` can be read against `train`.

One number needs a caveat carried with it. `train.compile` reads near-zero
whenever XLA's persistent cache is warm (see Caching in `CLAUDE.md`), which is
the common case and would point optimization effort at the wrong place.
`timings.json` therefore records whether the cache directory held anything when
the run started, sampled before the first compile could fill it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast, override

from rich.console import Console
from rich.table import Table

from .rantypes import COMPILE_CACHE_DIR

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path
    from typing import Any, LiteralString

TIMING_ENV_VAR: Final[LiteralString] = "RAN_TIMING"

# Spelled out rather than `bool(value)`, because the string "0" is truthy and a
# SLURM `--export` that forwards an unset variable delivers "" rather than
# absence -- the same trap `CACHE_ENV_VAR` documents.
_FALSEY: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})


def _enabled_from_env(environ: Mapping[str, str], /) -> bool:
    return environ.get(TIMING_ENV_VAR, "").strip().lower() not in _FALSEY


@dataclass(frozen=True, slots=True)
class Phase:
    """One completed phase. `depth` is 0 for a top-level phase."""

    name: str
    seconds: float
    depth: int
    detail: str | None = None
    failed: bool = False


class _Open:
    """The handle a live phase yields. One instance per open phase."""

    __slots__ = ("detail",)

    def __init__(self) -> None:
        self.detail: str | None = None

    def note(self, detail: str, /) -> None:
        """Annotate the phase, e.g. which branch of a cache check it took."""
        self.detail: str = detail

    def block[T](self, value: T, /) -> T:
        """Wait for JAX to finish producing `value`, inside the clock.

        JAX is async, so a timer stopped before the arrays are ready charges
        this phase's time to whichever phase runs next. This shifts *when* the
        wait happens and never what is computed; with timing off it does not
        happen at all, which is why it lives on the handle rather than at the
        call site.
        """
        import jax

        return cast(typ="T", val=jax.block_until_ready(x=value))


class _Noop(_Open):
    """The single instance handed out when timing is off."""

    __slots__ = ()

    @override
    def note(self, detail: str, /) -> None:
        del detail

    @override
    def block[T](self, value: T, /) -> T:
        return value


_NOOP: Final[_Noop] = _Noop()


class _Recorder:
    """Completed phases, in closing order, plus the live nesting depth."""

    __slots__ = ("compile_cache_warm", "depth", "names", "open", "records")

    def __init__(self) -> None:
        self.records: list[Phase] = []
        self.open: list[_Open] = []
        self.names: list[str] = []
        self.depth: int = 0
        # Sampled at construction -- before the run has had a chance to compile
        # anything into it.
        self.compile_cache_warm: bool = any(COMPILE_CACHE_DIR.glob(pattern="*"))


_recorder: _Recorder | None = _Recorder() if _enabled_from_env(os.environ) else None


def is_enabled() -> bool:
    return _recorder is not None


def enable(active: bool = True, /) -> None:
    """Turn timing on or off and discard anything already recorded.

    The environment decides this at import; this exists for tests and for a
    caller that knows better than the environment does.
    """
    global _recorder
    _recorder = _Recorder() if active else None


def phases() -> tuple[Phase, ...]:
    """Completed phases in closing order: children before their parent."""
    return () if _recorder is None else tuple(_recorder.records)


def note(detail: str, /, *, to: str | None = None) -> None:
    """Annotate an open phase, if there is one.

    This is what lets `datasets.py` say "cache hit" about a phase that
    `workflow.py` opened, without the loaders having to own a phase of their
    own or thread a handle down through their signatures.

    `to` names which open phase it means, and the loaders always pass it. They
    are called from more than one place --- `evaluate_run` rebuilds the same
    dataset inside the `evaluate` phase --- and annotating the innermost open
    phase would put "cache hit" in the detail column of a row about metrics.
    Unnamed, or named for a phase that is not open, the note is dropped.
    """
    if _recorder is None or not _recorder.open:
        return
    if to is None:
        _recorder.open[-1].note(detail)
        return
    for handle, name in zip(
        reversed(_recorder.open), reversed(_recorder.names), strict=True
    ):
        if name == to:
            handle.note(detail)
            return


@contextmanager
def phase(name: str, /, *, detail: str | None = None) -> Generator[_Open]:
    """Time the block, unless timing is off -- in which case do nothing at all."""
    recorder: _Recorder | None = _recorder
    if recorder is None:
        yield _NOOP
        return

    handle = _Open()
    handle.detail = detail
    depth: int = recorder.depth
    recorder.open.append(handle)
    recorder.names.append(name)
    recorder.depth = depth + 1
    start: float = time.perf_counter()
    failed = False
    try:
        yield handle
    except BaseException:
        # The phase that just fell over is the one whose number is most worth
        # having, so record it and re-raise rather than losing it.
        failed = True
        raise
    finally:
        elapsed: float = time.perf_counter() - start
        recorder.depth = depth
        _ = recorder.open.pop()
        _ = recorder.names.pop()
        recorder.records.append(
            Phase(
                name=name,
                seconds=elapsed,
                depth=depth,
                detail=handle.detail,
                failed=failed,
            )
        )


def _total_seconds(records: list[Phase], /) -> float:
    """Sum of the top-level phases only; a nested one is already inside its parent."""
    return sum(p.seconds for p in records if p.depth == 0)


def _ordered(records: list[Phase], /) -> list[Phase]:
    """Closing order puts children before parents. Read it back parents-first."""
    ordered: list[Phase] = []
    for i, record in enumerate(records):
        if record.depth != 0:
            continue
        ordered.append(record)
        # A top-level phase's children are the records that closed before it and
        # after the previous top-level one.
        start: int = next(
            (j + 1 for j in range(i - 1, -1, -1) if records[j].depth == 0), 0
        )
        ordered.extend(records[start:i])
    return ordered


def report(console: Console | None = None, /) -> None:
    """Print the phase table. Silent when timing is off or nothing was timed."""
    if _recorder is None or not _recorder.records:
        return
    total: float = _total_seconds(_recorder.records)
    table = Table(title="Wall clock by phase")
    table.add_column(header="Phase")
    table.add_column(header="Seconds", justify="right")
    table.add_column(header="Share", justify="right")
    table.add_column(header="Detail")
    for record in _ordered(_recorder.records):
        share: str = f"{100.0 * record.seconds / total:.1f}%" if total > 0 else "-"
        detail: str = record.detail or ""
        if record.failed:
            detail = f"{detail} (raised)".strip()
        table.add_row(
            "  " * record.depth + record.name,
            f"{record.seconds:.3f}",
            share,
            detail,
        )
    (console or Console()).print(table)


def write(run_dir: Path, /) -> None:
    """Write `timings.json`. A no-op when timing is off or nothing was timed.

    Flat, with a `depth` field rather than nested objects, so a sweep can join
    it against `config.json` without walking a tree. Every number here comes
    from `perf_counter`, so the `np.float32` JSON hazard `CLAUDE.md` warns about
    cannot arise --- nothing needs coercing on the way out.
    """
    if _recorder is None or not _recorder.records:
        return
    payload: dict[str, Any] = {
        "total_seconds": _total_seconds(_recorder.records),
        "compile_cache_warm": _recorder.compile_cache_warm,
        "phases": [
            {
                "name": p.name,
                "seconds": p.seconds,
                "depth": p.depth,
                "detail": p.detail,
                "failed": p.failed,
            }
            for p in _ordered(_recorder.records)
        ],
    }
    _ = (run_dir / "timings.json").write_text(data=json.dumps(obj=payload, indent=2))
