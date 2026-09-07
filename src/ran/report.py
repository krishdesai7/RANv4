"""Compile a run directory into one LaTeX dossier.

`config.json`, `metrics.json` and `timings.json` are machine interfaces and
stay exactly as they are; this module is a read-only consumer that turns them
into something a person reads. All rounding policy lives in the template, in
siunitx column specifications -- a column formatted string-by-string in Python
cannot align on the decimal marker -- so everything here emits full precision
and lets LaTeX decide how much of it to show.
"""

from __future__ import annotations

import json
import logging
import math
import re
from importlib import resources
from typing import TYPE_CHECKING, Any, Final

from .rantypes import JET_OBS, JET_VARIABLE_GROUPS, artifacts_dir

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger
    from pathlib import Path

logger: Logger = logging.getLogger(name=__name__)

# `<<[A-Z_]+>>` rather than a bare `<<`: the template's own header comment
# documents the replacement contract and legitimately contains `<<...>>`.
TEMPLATE_TOKEN: Final[re.Pattern[str]] = re.compile(pattern=r"<<[A-Z_]+>>")

_LATEX_SPECIALS: Final[dict[str, str]] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Enough places for four significant figures of the smallest metric a real run
# produces (~8.3e-5 -> 0.00008294). The template's D column is `table-format=2.8`.
_DECIMAL_PLACES: Final[int] = 10


def load_template() -> str:
    """The shipped LaTeX template, as text."""
    return (resources.files("ran") / "templates" / "report.tex").read_text(
        encoding="utf-8"
    )


def decimal(value: float, /) -> str:
    """Plain decimal notation, never exponential.

    Every numeric cell lands in a siunitx `S` column, which fixes a
    `table-format` and cannot absorb an exponent -- a value emitted as
    `8.294e-05` misaligns the column silently rather than erroring.
    """
    rendered: str = f"{value:.{_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def latex_text(value: str, /) -> str:
    """Escape the characters that would end a compile or change the meaning."""
    return "".join(_LATEX_SPECIALS.get(character, character) for character in value)


# Keys whose values span the full row (`\ConfigWide`) rather than pairing two
# entries per line (`\ConfigPair`): a variable list, a Gaussian's raw
# parameters, and the MMD bandwidth brackets, all of which are either too long
# or too structured for a single narrow cell.
_WIDE_KEYS: Final[frozenset[str]] = frozenset(
    {"variables", "mmd_sigmas_detector", "mmd_sigmas_particle", "gaussian_params"}
)


def _scalar_cell(value: object, /) -> str:
    r"""One `\ConfigPair` cell for a scalar config value."""
    if isinstance(value, bool):
        return latex_text(str(value))
    if isinstance(value, int):
        return rf"\Count{{{value}}}"
    if isinstance(value, float):
        return rf"\Raw{{{decimal(value)}}}"
    return rf"\ConfigVal{{{value}}}"


def _format_value(value: object, /) -> str:
    r"""Render a (possibly nested) JSON-ish value for a `\ConfigVal` cell."""
    if isinstance(value, list):
        items: list[object] = list(value)  # pyrefly: unknown element type from json.
        return "[" + ", ".join(_format_value(item) for item in items) + "]"
    if isinstance(value, float):
        return decimal(value)
    return str(value)


def _gaussian_params_cell(params: Mapping[str, Any], /) -> str:
    """`gaussian_params` -- a Gaussian run's `dim`/`mu_*`/`cov_*` -- as text.

    There is no scalar slot for a mean vector or a covariance matrix, so the
    whole dict is rendered `key=value` and spans the row like `variables` does
    for a jet run.
    """
    rendered: str = ", ".join(f"{k}={_format_value(v)}" for k, v in params.items())
    return rf"\ConfigVal{{{rendered}}}"


def _sigma_cell(sigmas: Sequence[float], /) -> str:
    r"""`median x (1/2 .. 2)` when the values really are the bracket.

    The five numbers recorded as `mmd_sigmas_detector`/`_particle` are not
    free parameters: they are `mmd.median_bandwidth` of the data side, scaled
    by `mmd._SCALES`. Printed raw they are five opaque floats. The recorded
    values are verified against that single source of truth before
    collapsing, so a future change to the scale set falls back to printing
    them in full rather than silently mislabelling a run.
    """
    # Deferred: `ran.mmd` imports jax, which this module must not load eagerly.
    from .mmd import _SCALES

    if len(sigmas) == len(_SCALES):
        median: float = sigmas[_SCALES.index(1.0)]
        if all(
            math.isclose(s, median * scale, rel_tol=1e-6)
            for s, scale in zip(sigmas, _SCALES, strict=True)
        ):
            return (
                rf"\Raw{{{decimal(median)}}} "
                r"$\times\ (1/2,\ 1/\sqrt2,\ 1,\ \sqrt2,\ 2)$"
            )
    return ", ".join(rf"\Raw{{{decimal(s)}}}" for s in sigmas)


def _pair_lines(scalars: list[tuple[str, Any]], /) -> list[str]:
    r"""`\ConfigPair` rows, two entries per line, the trailing odd one padded."""
    lines: list[str] = []
    for i in range(0, len(scalars), 2):
        pair: list[tuple[str, Any]] = scalars[i : i + 2]
        if len(pair) == 1:
            pair.append(("", ""))
        (k1, v1), (k2, v2) = pair
        c1: str = "" if k1 == "" else _scalar_cell(v1)
        c2: str = "" if k2 == "" else _scalar_cell(v2)
        lines.append(rf"\ConfigPair{{{k1}}}{{{c1}}}{{{k2}}}{{{c2}}}")
    return lines


def _wide_lines(entries: Mapping[str, Any], /) -> list[str]:
    r"""`\ConfigWide` rows: the variable list, Gaussian params, MMD sigmas."""
    lines: list[str] = []
    if "variables" in entries:
        names: str = ", ".join(entries["variables"])
        lines.append(rf"\ConfigWide{{variables}}{{\ConfigVal{{{names}}}}}")
    if "gaussian_params" in entries:
        cell: str = _gaussian_params_cell(entries["gaussian_params"])
        lines.append(rf"\ConfigWide{{gaussian_params}}{{{cell}}}")
    lines.extend(
        rf"\ConfigWide{{{key}}}{{{_sigma_cell(entries[key])}}}"
        for key in ("mmd_sigmas_detector", "mmd_sigmas_particle")
        if key in entries
    )
    return lines


def config_rows(config: Mapping[str, Any], timings: Mapping[str, Any] | None, /) -> str:
    """`<<CONFIG_ROWS>>`: two key/value pairs per line, wide values spanning.

    `compile_cache_warm` lives at the top level of `timings.json`, not in
    `config.json` -- it is folded in here as a config row because it is the
    fact that makes the `compile` timing interpretable at all.
    """
    entries: dict[str, Any] = dict(config)
    if timings is not None and "compile_cache_warm" in timings:
        entries["compile_cache_warm"] = timings["compile_cache_warm"]

    scalars: list[tuple[str, Any]] = [
        (k, v) for k, v in entries.items() if k not in _WIDE_KEYS
    ]
    return "\n".join(_pair_lines(scalars) + _wide_lines(entries))


def _phase_line(phase: Mapping[str, Any], total: float, /) -> str:
    """One timing-table row for a single recorded phase."""
    name: str = latex_text(phase["name"])
    depth: int = phase["depth"]
    seconds: float = float(phase["seconds"])
    if depth:
        name = rf"\TimingSubphase{{{name}}}"
    # A nested phase is already inside its parent's share, so its cell is
    # empty rather than a number that would not sum to a hundred.
    share: str = decimal(100.0 * seconds / total) if depth == 0 and total > 0 else ""
    # `pass` ("train"/"load") has no column of its own in the four-column
    # template; it is folded into Detail instead of widening the table. Older
    # `timings.json` files predate the field, so `.get` rather than `[...]`.
    parts: list[str] = [p for p in (phase.get("detail"), phase.get("pass")) if p]
    detail: str = latex_text(" -- ".join(parts))
    if phase.get("failed"):
        detail = rf"{detail} \textbf{{(raised)}}" if detail else r"\textbf{(raised)}"
    return f"{name} & {decimal(seconds)} & {share} & {detail} \\\\"


def timing_rows(timings: Mapping[str, Any], /) -> str:
    """`<<TIMINGS_ROWS>>`: one row per phase, a rule, then the bold total."""
    total: float = float(timings["total_seconds"])
    lines: list[str] = [_phase_line(phase, total) for phase in timings["phases"]]
    lines.extend((r"\midrule", rf"\textbf{{total}} & {decimal(total)} & 100 & \\"))
    return "\n".join(lines)


_DASH: Final[str] = r"\multicolumn{1}{c}{---}"
_METRICS: Final[tuple[str, ...]] = ("wasserstein", "jensenshannon", "triangular")


def _row(
    variable: str,
    level: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    daggered: bool,
    /,
) -> str:
    """One variable's sixteen cells: label, then Sim/IBU/IBU%/RAN/RAN% x 3."""
    symbol: str = (
        JET_OBS[variable].symbol if variable in JET_OBS else latex_text(variable)
    )
    label: str = rf"{symbol}$^\dag$" if daggered else symbol
    ours: Mapping[str, float] = ran[f"{level}_{variable}"]
    theirs: Mapping[str, float] | None = (
        ibu.get(f"{level}_{variable}") if ibu is not None else None
    )

    cells: list[str] = [label]
    for metric in _METRICS:
        cells.append(decimal(ours[f"{metric}_before"]))
        if theirs is None:
            cells.extend((_DASH, _DASH))
        else:
            cells.extend(
                (
                    decimal(theirs[f"{metric}_after"]),
                    decimal(theirs[f"{metric}_improvement_pct"]),
                )
            )
        cells.extend(
            (
                decimal(ours[f"{metric}_after"]),
                decimal(ours[f"{metric}_improvement_pct"]),
            )
        )
    return " & ".join(cells) + r" \\"


def _group_lines(
    label: str,
    members: Sequence[str],
    level: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> list[str]:
    """A rule, an italic heading spanning all 16 columns, then its rows."""
    return [
        r"\midrule",
        rf"\multicolumn{{16}}{{@{{}}l}}{{\itshape {label}}} \\",
        *(_row(v, level, ran, ibu, v in skipped) for v in members),
    ]


def metrics_table(
    level: str,
    variables: Sequence[str],
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> str:
    """`<<DETECTOR_TABLE>>` / `<<PARTICLE_TABLE>>`: the row bodies only.

    The template owns the tabular, the column specification and the header;
    this owns the rules, the group headings and the data rows. `skipped` names
    the variables IBU gave up on, which are marked rather than shown as an
    honest-looking 0.0% improvement.
    """
    present: frozenset[str] = frozenset(variables)
    lines: list[str] = [
        line
        for label, members in JET_VARIABLE_GROUPS
        if (in_group := [v for v in members if v in present])
        for line in _group_lines(label, in_group, level, ran, ibu, skipped)
    ]

    if not lines:  # a non-jet run: rows, no grouping
        lines.append(r"\midrule")
        lines.extend(_row(v, level, ran, ibu, v in skipped) for v in variables)
    return "\n".join(lines)


def skipped_variables(
    run_dir: Path, ibu: Mapping[str, Any] | None, /
) -> frozenset[str]:
    """Variables IBU's purity binning refused, from the recorded outcomes.

    Falls back to the observable signature -- an `after` exactly equal to its
    `before` -- for a `metrics_ibu.json` written before outcomes were recorded.
    """
    path: Path = artifacts_dir(run_dir) / "ibu_outcomes.json"
    try:
        outcomes: list[dict[str, Any]] = json.loads(path.read_text())
    except OSError, ValueError:
        if ibu is None:
            return frozenset()
        return frozenset(
            key.split("_", 1)[1]
            for key, entry in ibu.items()
            if entry["wasserstein_after"] == entry["wasserstein_before"]
        )
    return frozenset(o["variable_name"] for o in outcomes if o["status"] == "skipped")
