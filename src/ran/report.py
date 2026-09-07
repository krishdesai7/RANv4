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
import shutil

# One fixed argv, no shell, and the only interpolated element is a path this
# process just wrote; see `_compile`.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from importlib import resources
from typing import TYPE_CHECKING, Any, Final, cast

from .rantypes import ARTIFACTS_DIR, JET_OBS, JET_VARIABLE_GROUPS, artifacts_dir

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


# A daggered label is meaningless without this line. IBU returning its input
# unchanged records an "after" bit-identical to its "before" and a 0.0%
# improvement, which reads as "IBU tried and achieved nothing"; the truth is
# that it declined to unfold the observable at all. The row spans all sixteen
# columns and sits immediately before the template's `\bottomrule`.
_DAGGER_LEGEND: Final[str] = (
    r"\multicolumn{16}{@{}l}{\footnotesize $^\dag$ IBU's purity binning "
    r"produced fewer than two bins for this observable, so IBU declined to "
    r"unfold it and returned its input unchanged. The improvement shown for "
    r"it is not a measurement.} \\"
)


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


def _populated_groups(present: frozenset[str], /) -> list[tuple[str, Sequence[str]]]:
    """The display groups this run's variables actually populate, in order.

    `--var m --var w` leaves the splitting group empty, and an empty group
    would print a heading with no rows under it.
    """
    return [
        (label, in_group)
        for label, members in JET_VARIABLE_GROUPS
        if (in_group := [v for v in members if v in present])
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
    honest-looking 0.0% improvement; when any of them lands in this table the
    body ends with a legend row explaining the mark.
    """
    groups: list[tuple[str, Sequence[str]]] = _populated_groups(frozenset(variables))
    lines: list[str] = [
        line
        for label, members in groups
        for line in _group_lines(label, members, level, ran, ibu, skipped)
    ]
    emitted: list[str] = [v for _, members in groups for v in members]

    if not lines:  # a non-jet run: rows, no grouping
        emitted = list(variables)
        lines.append(r"\midrule")
        lines.extend(_row(v, level, ran, ibu, v in skipped) for v in emitted)

    # Only when the mark is actually on the page: an unexplained legend is as
    # confusing as an unexplained dagger.
    if any(v in skipped for v in emitted):
        lines.append(_DAGGER_LEGEND)
    return "\n".join(lines)


def skipped_variables(
    run_dir: Path, ibu: Mapping[str, Any] | None, /
) -> frozenset[str]:
    """Variables IBU's purity binning refused, from the recorded outcomes.

    Falls back to the observable signature -- an `after` exactly equal to its
    `before` -- for a `metrics_ibu.json` written before outcomes were recorded.
    """
    # `run_dir / ARTIFACTS_DIR` rather than `artifacts_dir(run_dir)`: the
    # latter creates the directory, and this module only ever reads. Pointed
    # at a directory that is not a run, `ran report` must fail without
    # littering it.
    path: Path = run_dir / ARTIFACTS_DIR / "ibu_outcomes.json"
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


# A run that died before `ran evaluate` still deserves a report: the tables
# degrade to a single explanatory row rather than raising. The template fixes
# sixteen columns, so the row has to span all of them.
_NO_METRICS: Final[str] = (
    r"\midrule"
    "\n"
    r"\multicolumn{16}{@{}l}{\itshape metrics.json not found: "
    r"run \texttt{ran evaluate} for this run.} \\"
)


def _read(path: Path, /) -> dict[str, Any] | None:
    """One JSON artifact, or `None` when it is absent or unreadable."""
    try:
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return None


def _variables(config: Mapping[str, Any], /) -> tuple[str, ...]:
    """The run's column names: recorded for a jet run, positional otherwise."""
    recorded: object = config.get("variables")
    if recorded:
        return tuple(cast("Sequence[str]", recorded))
    return tuple(f"dim_{i}" for i in range(int(config["dim"])))


def _table(
    level: str,
    variables: Sequence[str],
    ran: Mapping[str, Any] | None,
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> str:
    """A metrics body, or the not-found row when there are no metrics."""
    if not ran:
        return _NO_METRICS
    return metrics_table(level, variables, ran, ibu, skipped)


def render(run_dir: Path, /) -> str:
    """The fully substituted LaTeX source for one run directory."""
    # There is no run without a config, so this is the one hard error: every
    # other artifact degrades to dashes or a labelled row.
    config_path: Path = run_dir / "config.json"
    config: dict[str, Any] | None = _read(config_path)
    if config is None:
        state: str = "is unreadable" if config_path.exists() else "does not exist"
        msg: str = f"{config_path} {state}: not a run directory"
        raise FileNotFoundError(msg)

    # `run_dir / ARTIFACTS_DIR` rather than `artifacts_dir(...)`: rendering
    # reads, and must not create a directory in something that is not a run.
    artifacts: Path = run_dir / ARTIFACTS_DIR
    ran: dict[str, Any] | None = _read(artifacts / "metrics.json")
    ibu: dict[str, Any] | None = _read(artifacts / "metrics_ibu.json")
    timings: dict[str, Any] | None = _read(artifacts / "timings.json")
    skipped: frozenset[str] = skipped_variables(run_dir, ibu)
    variables: tuple[str, ...] = _variables(config)

    source: str = load_template()
    for token, value in (
        ("<<RUN_NAME>>", run_dir.name),
        ("<<CONFIG_ROWS>>", config_rows(config, timings)),
        ("<<TIMINGS_ROWS>>", timing_rows(timings) if timings else ""),
        ("<<DETECTOR_TABLE>>", _table("detector", variables, ran, ibu, skipped)),
        ("<<PARTICLE_TABLE>>", _table("particle", variables, ran, ibu, skipped)),
        # Absolute: `pdflatex` runs in `artifacts/`, so a relative path would
        # not resolve, and a sweep arm's directory name (`lrg1e-4_seed03`)
        # cannot be reconstructed from a bare basename either.
        ("<<FIGURE_DIR>>", str(artifacts.resolve())),
    ):
        source = source.replace(token, value)

    left: list[str] = TEMPLATE_TOKEN.findall(source)
    if left:
        msg = f"template tokens with no value: {', '.join(sorted(set(left)))}"
        raise ValueError(msg)
    return source


_LATEX_ARGS: Final[tuple[str, ...]] = (
    "pdflatex",
    "-interaction=nonstopmode",
    "-halt-on-error",
)
# Kept on failure so the compile can be debugged; removed on success so the run
# root holds only `report.pdf` and `config.json`.
_AUX_SUFFIXES: Final[tuple[str, ...]] = (".aux", ".log", ".out")


def _compile(source: Path, artifacts: Path, run_dir: Path, /) -> None:
    r"""Run `pdflatex` twice, from `artifacts/`, emitting into the run root.

    Twice because the first pass cannot know the `\includegraphics` box sizes
    or the final page count, and the header's page number depends on both.
    """
    if shutil.which("pdflatex") is None:
        msg: str = (
            "pdflatex is not on PATH. Install a TeX distribution, or pass "
            "--no-compile to emit report.tex alone."
        )
        raise RuntimeError(msg)

    for _pass in range(2):
        # Fixed argv, no shell, and the only interpolated element is a path
        # this process just wrote.
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [*_LATEX_ARGS, f"-output-directory={run_dir}", source.name],
            cwd=artifacts,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail: str = "\n".join(completed.stdout.splitlines()[-40:])
            msg = f"pdflatex failed for {run_dir.name}:\n{tail}"
            raise RuntimeError(msg)


def build_report(
    run_dir: Path, /, *, force: bool = False, compile_pdf: bool = True
) -> Path:
    """Write `artifacts/report.tex`, compile `report.pdf` at the run root.

    Returns what it produced: the PDF, or the LaTeX source under
    `--no-compile`.
    """
    pdf: Path = run_dir / "report.pdf"
    if compile_pdf and pdf.exists() and not force:
        logger.info("%s: report.pdf exists, skipping (use --force)", run_dir.name)
        return pdf

    source: str = render(run_dir)
    tex: Path = artifacts_dir(run_dir) / "report.tex"
    _ = tex.write_text(source, encoding="utf-8")
    if not compile_pdf:
        logger.info("%s: saved %s", run_dir.name, tex)
        return tex

    _compile(tex, tex.parent, run_dir)
    for suffix in _AUX_SUFFIXES:
        (run_dir / f"report{suffix}").unlink(missing_ok=True)
    logger.info("%s: saved %s", run_dir.name, pdf)
    return pdf
