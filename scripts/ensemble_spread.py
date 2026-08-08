#!/usr/bin/env python3
"""Spread across an ensemble of RAN runs that differ along one axis.

    # one ensemble: mean, SD and range per metric
    uv run scripts/ensemble_spread.py bench/ensemble_jax_init_<ts>

    # place a single run against that spread
    uv run scripts/ensemble_spread.py bench/ensemble_jax_init_<ts> \\
        --reference=bench/tf_<ts>/run_artifacts/metrics.json

    # two ensembles: Welch's t-test, unequal n and unequal variance
    uv run scripts/ensemble_spread.py bench/ensemble_jax_init_<ts> \\
        --compare=bench/ensemble_tf_init_<ts>

Reads the metrics_member*.json that the submit_ensemble scripts copy out of
each member. The SD across members is the model uncertainty along whichever
axis that ensemble varied -- weight init, or the train/val/test partition.

--reference treats the single run as a fixed point, which is only honest when
its own spread is known to be small. --compare does not: it propagates both
SDs, so it is the right tool for asking whether two arms actually differ.

What neither can do: the arms partition the data differently, so a difference
between them confounds implementation with split. Size that confound with a
VARY=data ensemble and compare the two SDs before reading anything causal into
a cross-arm gap.

Stdlib-only but for an optional scipy import behind a try, so this runs on
either branch. Importing `ran` would pin a Keras backend; scipy will not.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# One member's metrics.json: variable -> metric name -> value.
type Metrics = dict[str, dict[str, float]]

FAMILIES = ("wasserstein", "jensenshannon", "triangular")
# metrics_seed*.json is the older spelling, kept so early ensembles still read.
MEMBER_RE = re.compile(r"metrics_(?:member|seed)(\d+)\.json$")
# Beyond this many standard deviations, a difference is not plausibly the same
# quantity measured again.
FLAG_SD = 2.0


def _out(line: str = "") -> None:
    """The report is this tool's product, so it goes to stdout deliberately.

    Not `print`: ruff's T20 and tests/test_source_hygiene.py both ban the
    builtin under scripts/. This is the named channel they ask for.
    """
    sys.stdout.write(f"{line}\n")


def _err(line: str) -> None:
    sys.stderr.write(f"{line}\n")


def _load_members(bench_dir: Path) -> dict[int, Metrics]:
    """Map member index -> that member's metrics.json."""
    members: dict[int, Metrics] = {}
    for path in sorted(bench_dir.glob("metrics_*.json")):
        match = MEMBER_RE.search(path.name)
        if match is None:
            continue
        try:
            members[int(match.group(1))] = json.loads(path.read_text())
        except json.JSONDecodeError:
            _err(f"skipping unreadable {path}")
    return members


def _provenance(bench_dir: Path) -> dict[str, str]:
    path = bench_dir / "provenance.txt"
    if not path.exists():
        return {}
    pairs = (
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )
    return {k: v for k, v in pairs}


def _label(bench_dir: Path) -> str:
    prov = _provenance(bench_dir)
    arm = prov.get("arm", bench_dir.name)
    vary = prov.get("vary")
    return f"{arm} (vary={vary})" if vary else arm


def _variables(members: dict[int, Metrics]) -> list[str]:
    """Variables every member reported, so no column is a ragged average."""
    common: set[str] | None = None
    for metrics in members.values():
        keys = set(metrics)
        common = keys if common is None else common & keys
    return sorted(common or ())


def _values(members: dict[int, Metrics], variable: str, key: str) -> list[float]:
    return [
        float(metrics[variable][key])
        for metrics in members.values()
        if key in metrics.get(variable, {})
    ]


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def _metric_keys(raw: bool) -> list[str]:
    keys = [f"{f}_improvement_pct" for f in FAMILIES]
    if raw:
        keys += [f"{f}_after" for f in FAMILIES]
    return keys


# ------------------------------------------------------------------ one ensemble


def _verdict(reference: float, values: list[float]) -> str:
    """Where a single reference value sits relative to the ensemble."""
    lo, hi = min(values), max(values)
    if lo <= reference <= hi:
        return "within spread"
    if len(values) < 2:
        return "n=1, no spread"
    sd = statistics.stdev(values)
    if sd <= 0.0:
        return "identical members, outside"
    z = (reference - statistics.fmean(values)) / sd
    return f"{z:+.1f} sd" + ("  <<" if abs(z) > FLAG_SD else "")


def _row(variable: str, values: list[float], reference: float | None) -> str:
    row = (
        f"  {variable:<18}"
        f"{statistics.fmean(values):>10.3f}"
        f"{_sd(values):>9.3f}"
        f"{min(values):>10.3f}"
        f"{max(values):>10.3f}"
    )
    if reference is None:
        return row
    return f"{row}{reference:>10.3f}   {_verdict(reference, values)}"


def _spread_table(
    members: dict[int, Metrics],
    variables: list[str],
    key: str,
    reference: Metrics | None,
) -> None:
    _out(f"{key}")
    header = f"  {'VARIABLE':<18}{'MEAN':>10}{'SD':>9}{'MIN':>10}{'MAX':>10}"
    _out(header + (f"{'REF':>10}   DEVIATION" if reference else ""))
    for variable in variables:
        values = _values(members, variable, key)
        if not values:
            continue
        ref_value = None
        if reference is not None and key in reference.get(variable, {}):
            ref_value = float(reference[variable][key])
        _out(_row(variable, values, ref_value))
    _out()


# ----------------------------------------------------------------- two ensembles


def _p_value(a: list[float], b: list[float]) -> float | None:
    """Welch's two-sided p, or None when scipy is unavailable.

    Welch rather than Student: the arms have different n and no reason to
    share a variance -- the faster arm is cheap enough to over-sample.
    """
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(stats.ttest_ind(b, a, equal_var=False).pvalue)


def _welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """(difference, standard error) for b - a."""
    diff = statistics.fmean(b) - statistics.fmean(a)
    if len(a) < 2 or len(b) < 2:
        return diff, float("nan")
    var_a = statistics.variance(a) / len(a)
    var_b = statistics.variance(b) / len(b)
    return diff, (var_a + var_b) ** 0.5


def _compare_row(variable: str, a: list[float], b: list[float]) -> str:
    diff, se = _welch(a, b)
    row = (
        f"  {variable:<18}"
        f"{statistics.fmean(a):>8.2f} ± {_sd(a):<6.2f}"
        f"{statistics.fmean(b):>8.2f} ± {_sd(b):<6.2f}"
        f"{diff:>9.2f}"
    )
    if not se > 0.0:
        return f"{row}{'--':>10}"
    sigma = diff / se
    p = _p_value(a, b)
    p_text = "  (no scipy)" if p is None else f"{p:>11.2g}"
    return f"{row}{sigma:>9.1f}{p_text}" + ("  <<" if abs(sigma) > FLAG_SD else "")


def _compare_table(
    a_members: dict[int, Metrics],
    b_members: dict[int, Metrics],
    variables: list[str],
    key: str,
) -> None:
    _out(f"{key}")
    _out(
        f"  {'VARIABLE':<18}{'A mean ± sd':>17}{'B mean ± sd':>17}"
        f"{'B-A':>9}{'SIGMA':>9}{'p':>11}"
    )
    for variable in variables:
        a = _values(a_members, variable, key)
        b = _values(b_members, variable, key)
        if not a or not b:
            continue
        _out(_compare_row(variable, a, b))
    _out()


# ------------------------------------------------------------------------ driver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_dir", type=Path, help="bench/ensemble_<arm>_<vary>_<ts>")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a single run's metrics.json to place against the spread",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="another ensemble bench dir; runs Welch's t-test against it",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="also tabulate the _after values, not just _improvement_pct",
    )
    return parser.parse_args()


def _run_spread(args: argparse.Namespace, members: dict[int, Metrics]) -> int:
    reference = None
    if args.reference is not None:
        if not args.reference.exists():
            _err(f"reference not found: {args.reference}")
            return 1
        reference = json.loads(args.reference.read_text())

    ids = ", ".join(str(s) for s in sorted(members))
    _out("=" * 78)
    _out(f"ensemble spread: {_label(args.bench_dir)}, {len(members)} members ({ids})")
    _out(f"source: {args.bench_dir}")
    if reference is not None:
        _out(f"reference: {args.reference}")
    if len(members) < 3:
        _out("NOTE: fewer than 3 members -- the SD is not yet meaningful.")
    _out("=" * 78)
    _out()

    variables = _variables(members)
    for key in _metric_keys(args.raw):
        _spread_table(members, variables, key, reference)
    return 0


def _run_compare(args: argparse.Namespace, b_members: dict[int, Metrics]) -> int:
    a_members = _load_members(args.compare)
    if not a_members:
        _err(f"no member metrics under {args.compare}")
        return 1

    _out("=" * 78)
    _out("two-sample comparison (Welch, two-sided)")
    _out(f"  A: {_label(args.compare)}  n={len(a_members)}   {args.compare}")
    _out(f"  B: {_label(args.bench_dir)}  n={len(b_members)}   {args.bench_dir}")
    _out(f"  '<<' marks |B-A| above {FLAG_SD:g} standard errors.")
    _out("  p is per metric and uncorrected; the rows below are far from")
    _out("  independent, so read the pattern across them, not one p-value.")
    _out("=" * 78)
    _out()

    variables = [v for v in _variables(b_members) if v in _variables(a_members)]
    for key in _metric_keys(args.raw):
        _compare_table(a_members, b_members, variables, key)
    return 0


def main() -> int:
    args = _parse_args()
    members = _load_members(args.bench_dir)
    if not members:
        _err(f"no member metrics under {args.bench_dir}")
        return 1
    if args.compare is not None:
        return _run_compare(args, members)
    return _run_spread(args, members)


if __name__ == "__main__":
    raise SystemExit(main())
