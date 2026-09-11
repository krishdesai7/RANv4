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

from .rantypes import (
    ARTIFACTS_DIR,
    JET_OBS,
    JET_VARIABLE_GROUPS,
    artifacts_dir,
    display_order,
    figure_pages,
)

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
# It is also a hard resolution floor: anything below ~5e-11 rounds away entirely
# and renders as the string "0" rather than as a small number.
_DECIMAL_PLACES: Final[int] = 10

# What a numeric cell shows when it has no number to show. `\multicolumn`
# because the target is a siunitx `S` column, which will not typeset prose.
_DASH: Final[str] = r"\multicolumn{1}{c}{---}"


def load_template() -> str:
    """The shipped LaTeX template, as text."""
    return (resources.files(anchor="ran") / "templates" / "report.tex").read_text(
        encoding="utf-8"
    )


def _plain(value: float, /) -> str:
    """Fixed-point text for a finite value; no finiteness check of its own."""
    rendered: str = f"{value:.{_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def decimal(value: float, /) -> str:
    """Plain decimal notation, never exponential; `_DASH` when not finite.

    Every numeric cell lands in a siunitx `S` column, which fixes a
    `table-format` and cannot absorb an exponent -- a value emitted as
    `8.294e-05` misaligns the column silently rather than erroring.

    The contract is therefore "a numeric *cell*", not "a bare number".
    siunitx treats `nan` and `inf` as a hard error rather than typesetting
    them, so one non-finite value -- reachable through `*_improvement_pct`
    whenever a `before` is 0.0 -- would kill the whole document. The guard
    lives here rather than in the row builders because every numeric cell in
    the report is emitted through this one function: a per-caller check would
    be a rule to remember at each new call site instead of one held in place.
    """
    if not math.isfinite(value):
        return _DASH
    return _plain(value)


def _num(macro: str, value: float, /) -> str:
    r"""`\<macro>{n}`, or a bare dash when there is no number to wrap.

    `\Raw` and `\Count` expand to siunitx `\num`, which rejects a non-finite
    argument exactly as an `S` column does, so the dash has to *replace* the
    wrapper rather than sit inside it.
    """
    rendered: str = decimal(value)
    return rendered if rendered == _DASH else rf"\{macro}{{{rendered}}}"


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
        return _num("Raw", value)
    return rf"\ConfigVal{{{value}}}"


def _format_value(value: object, /) -> str:
    r"""Render a (possibly nested) JSON-ish value for a `\ConfigVal` cell."""
    if isinstance(value, list):
        items: list[object] = list(value)  # pyrefly: unknown element type from json.
        return "[" + ", ".join(_format_value(item) for item in items) + "]"
    if isinstance(value, float):
        # `_plain`, not `decimal`: this lands inside `\ConfigVal`, which is
        # `\texttt{\detokenize{...}}` -- running text, where a `\multicolumn`
        # cell would be exactly the error the dash exists to avoid.
        return _plain(value) if math.isfinite(value) else str(value)
    return str(object=value)


def _variables_cell(names: Sequence[str], /) -> str:
    r"""The observable list as physics symbols, in display order.

    `config.json` records the column order as code identifiers (`tau21`,
    `f_ch`, `sdm`), which is right for a machine interface and wrong for a
    table a physicist reads -- the same row in the metrics tables already says
    `$\tau_{21}$`. Anything without a `JET_OBS` entry (a Gaussian run's
    `dim_0`) falls back to its escaped name.
    """
    ordered: tuple[int, ...] = display_order(names)
    return ", ".join(
        JET_OBS[name].symbol if name in JET_OBS else latex_text(name)
        for name in (names[i] for i in ordered)
    )


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
            math.isclose(a=s, b=median * scale, rel_tol=1e-6)
            for s, scale in zip(sigmas, _SCALES, strict=True)
        ):
            return (
                f"{_num('Raw', median)} "
                r"\(    \times\ (\frac12,\ \frac1{\sqrt2},\ 1,\ \sqrt2,\ 2)\)"
            )
    return ", ".join(_num("Raw", s) for s in sigmas)


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
        # Symbols, not `\ConfigVal`: the cell is math, not a detokenized
        # identifier list.
        lines.append(
            rf"\ConfigWide{{variables}}{{{_variables_cell(entries['variables'])}}}"
        )
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


# What each metric's column is multiplied by before printing, so all three
# tables read in units of 10^-3 and one exponent covers the report.
#
# Raw, a real twelve-observable run spans 6.2e-3..3.0e-1 (Wasserstein) and
# 8.5e-5..1.3e-2 (JS) -- columns of leading zeros. Scaled they read 6.2..303
# and 0.085..12.8. `triangular` is NOT scaled here because
# `evaluate._triangular_from_histograms` already multiplies by 1e3 on the way
# into `metrics.json`; scaling it again would misstate it by three orders of
# magnitude, which is exactly the error the header exists to prevent.
_SCALE: Final[dict[str, float]] = {
    "wasserstein": 1e3,
    "jensenshannon": 1e3,
    "triangular": 1.0,
}

# Column groups, in the order the report presents them. The token name is what
# `render` substitutes; the label is the table's own heading.
_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("WASSERSTEIN", "wasserstein"),
    ("JS", "jensenshannon"),
    ("VLC", "triangular"),
)


def _row(
    variable: str,
    level: str,
    metric: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    daggered: bool,
    /,
) -> str:
    """One variable's six cells for one metric: label, Sim, IBU, IBU%, RAN, RAN%.

    Six columns rather than the sixteen of a combined table: three metrics
    side by side needed `adjustbox` to shrink the whole thing to 7pt, which is
    below what anyone reads. Split, each table sets at full size.
    """
    symbol: str = (
        JET_OBS[variable].symbol if variable in JET_OBS else latex_text(variable)
    )
    label: str = rf"{symbol}\(^\dag\)" if daggered else symbol
    scale: float = _SCALE[metric]
    ours: Mapping[str, float] = ran[f"{level}_{variable}"]
    theirs: Mapping[str, float] | None = (
        ibu.get(f"{level}_{variable}") if ibu is not None else None
    )

    cells: list[str] = [label, decimal(ours[f"{metric}_before"] * scale)]
    if theirs is None:
        cells.extend((_DASH, _DASH))
    else:
        cells.extend(
            (
                decimal(theirs[f"{metric}_after"] * scale),
                # Improvements are ratios: scaling them would be wrong.
                decimal(theirs[f"{metric}_improvement_pct"]),
            )
        )
    cells.extend(
        (
            decimal(ours[f"{metric}_after"] * scale),
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
    r"\multicolumn{6}{@{}l}{\footnotesize \(^\dag\) IBU's purity binning "
    r"produced a single bin, so IBU failed to unfold.} \\"
)


def _group_lines(
    label: str,
    members: Sequence[str],
    level: str,
    metric: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> list[str]:
    """A rule, an upright bold heading spanning the table, then its rows.

    Upright rather than italic: a whole line of italicised text reads as an
    aside, and these headings are structure. Bold carries the same weight
    without the slant.
    """
    return [
        r"\midrule",
        rf"\multicolumn{{6}}{{@{{}}l}}{{\bfseries {label}}} \\",
        *(_row(v, level, metric, ran, ibu, v in skipped) for v in members),
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
    metric: str,
    variables: Sequence[str],
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> str:
    """One level's row bodies for one metric.

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
        for line in _group_lines(label, members, level, metric, ran, ibu, skipped)
    ]
    emitted: list[str] = [v for _, members in groups for v in members]

    if not lines:  # a non-jet run: rows, no grouping
        emitted = list(variables)
        lines.append(r"\midrule")
        lines.extend(_row(v, level, metric, ran, ibu, v in skipped) for v in emitted)

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
    r"\multicolumn{6}{@{}l}{\itshape metrics.json not found: "
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
    metric: str,
    variables: Sequence[str],
    ran: Mapping[str, Any] | None,
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> str:
    """A metrics body, or the not-found row when there are no metrics."""
    if not ran:
        return _NO_METRICS
    return metrics_table(level, metric, variables, ran, ibu, skipped)


def _figure_pages(artifacts: Path, stem: str, dim: int, /) -> str:
    r"""One `\ReportGraphic` block per page of a paginated level figure.

    `plotting._plot_level` writes the pages of one multi-page PDF, six panels
    each. The count is `figure_pages(dim)` rather than something read off the
    file, which keeps `report.py` free of a PDF dependency and free of
    matplotlib -- the two agree because they share the constant.
    """
    path: str = str(artifacts.resolve() / f"{stem}.pdf")
    return "\n\\clearpage\n".join(
        rf"\ReportPage{{{path}}}{{{page}}}" for page in range(1, figure_pages(dim) + 1)
    )


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
        *(
            (
                f"<<{level.upper()}_{token}>>",
                _table(level, metric, variables, ran, ibu, skipped),
            )
            for level in ("detector", "particle")
            for token, metric in _METRICS
        ),
        (
            "<<DETECTOR_FIGURES>>",
            _figure_pages(artifacts, "detector_level", len(variables)),
        ),
        (
            "<<PARTICLE_FIGURES>>",
            _figure_pages(artifacts, "particle_level", len(variables)),
        ),
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
            # `-interaction=nonstopmode` is set inside the document processor
            # and does not cover pdflatex's *pre-mode* prompts -- "I can't
            # write on file `report.log'" is asked before the mode takes
            # effect. Without this, a failing compile reads the parent's stdin
            # and an interactive `ran report` hangs instead of erroring.
            stdin=subprocess.DEVNULL,
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

    Resolved once, here, so every path downstream is absolute. `_compile` runs
    `pdflatex` with `cwd=artifacts` and hands it `-output-directory`, so a
    relative `run_dir` -- which is what `scripts/submit.sh` and the documented
    `ran report runs/<timestamp>` both pass -- would resolve against
    `artifacts/` instead of the caller's working directory.
    """
    run_dir = run_dir.resolve()
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
