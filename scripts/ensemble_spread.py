#!/usr/bin/env python3
"""Spread across an ensemble of RAN runs that differ only in weight init.

    uv run scripts/ensemble_spread.py bench/ensemble_jax_<ts>
    uv run scripts/ensemble_spread.py bench/ensemble_jax_<ts> \\
        --reference=bench/tf_<ts>/run_artifacts/metrics.json

Reads the metrics_seed*.json that scripts/submit_ensemble.sh copies out of each
member and reports mean, sample standard deviation and observed range per
metric. That standard deviation is the model uncertainty from initialization,
which is what tells you whether a single run's number is an effect or noise.

With --reference, a value from some other run is placed against that spread.
Use it to ask whether the two arms actually disagree. It answers only that
question: the reference is one run of a different implementation, so a value
outside the spread means "not explained by init variance alone" -- not that the
difference has been localized to any particular cause.

Deliberately stdlib-only, for the same reason as scripts/compare_runs.py:
importing `ran` would pin a Keras backend, and this has to run anywhere.
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
SEED_RE = re.compile(r"metrics_seed(\d+)\.json$")
# Beyond this many standard deviations from the ensemble mean, a reference
# value is not plausibly the same quantity measured again.
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
    """Map seed -> that member's metrics.json."""
    members: dict[int, Metrics] = {}
    for path in sorted(bench_dir.glob("metrics_seed*.json")):
        match = SEED_RE.search(path.name)
        if match is None:
            continue
        try:
            members[int(match.group(1))] = json.loads(path.read_text())
        except json.JSONDecodeError:
            _err(f"skipping unreadable {path}")
    return members


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


def _verdict(reference: float, values: list[float]) -> str:
    """Where the reference sits relative to the ensemble."""
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
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    row = (
        f"  {variable:<18}"
        f"{statistics.fmean(values):>10.3f}"
        f"{sd:>9.3f}"
        f"{min(values):>10.3f}"
        f"{max(values):>10.3f}"
    )
    if reference is None:
        return row
    return f"{row}{reference:>10.3f}   {_verdict(reference, values)}"


def _family_table(
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_dir", type=Path, help="bench/ensemble_<arm>_<ts>")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a metrics.json to place against the spread",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="also tabulate the _after values, not just _improvement_pct",
    )
    return parser.parse_args()


def _print_header(
    bench_dir: Path, members: dict[int, Metrics], reference_path: Path | None
) -> None:
    seeds = ", ".join(str(s) for s in sorted(members))
    _out("=" * 78)
    _out(f"ensemble spread: {len(members)} members (seeds {seeds})")
    _out(f"source: {bench_dir}")
    if reference_path is not None:
        _out(f"reference: {reference_path}")
    if len(members) < 3:
        _out("NOTE: fewer than 3 members -- the SD is not yet meaningful.")
    _out("=" * 78)
    _out()


def main() -> int:
    args = _parse_args()
    members = _load_members(args.bench_dir)
    if not members:
        _err(f"no metrics_seed*.json under {args.bench_dir}")
        return 1
    if args.reference is not None and not args.reference.exists():
        _err(f"reference not found: {args.reference}")
        return 1
    reference = (
        json.loads(args.reference.read_text()) if args.reference is not None else None
    )

    _print_header(args.bench_dir, members, args.reference)

    variables = _variables(members)
    keys = [f"{f}_improvement_pct" for f in FAMILIES]
    if args.raw:
        keys += [f"{f}_after" for f in FAMILIES]
    for key in keys:
        _family_table(members, variables, key, reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
