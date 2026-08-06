#!/usr/bin/env python3
"""Diff two benchmarked RAN runs: the JAX arm against the TensorFlow arm.

    uv run scripts/compare_runs.py bench/tf_<ts> bench/jax_<ts>

Takes the two bench directories written by scripts/submit_previous.sh and
scripts/submit.sh. The first is the baseline, the second the candidate; speedups
and metric deltas read "candidate relative to baseline".

Deliberately stdlib-only. It has to run on either branch and on a login node
without touching the project environment -- importing `ran` would pull in keras
and pin a backend, which is exactly what this comparison is about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Metric families written by ran.evaluate, in the order worth reading.
FAMILIES = ("wasserstein", "jensenshannon", "triangular")
STAGES = ("train", "omnifold", "ibu", "replot")


def _out(line: str = "") -> None:
    """The report is this tool's product, so it goes to stdout deliberately.

    Not `print`: ruff's T20 and tests/test_source_hygiene.py both ban the
    builtin under scripts/, on the grounds that output should go through a
    named channel rather than scattered prints. This is that channel.
    """
    sys.stdout.write(f"{line}\n")


def _err(line: str) -> None:
    sys.stderr.write(f"{line}\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_provenance(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            out[key.strip()] = value.strip()
    return out


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _fmt_gb(kb: float | None) -> str:
    return "--" if kb is None else f"{kb / 1024 / 1024:.2f}G"


def _ratio(base: float | None, cand: float | None) -> str:
    """Speedup of candidate over baseline. >1 means the candidate is faster."""
    if not base or not cand:
        return "--"
    return f"{base / cand:.2f}x"


def _artifact_dir(bench_dir: Path) -> Path:
    """Where the run's metrics live.

    The submit scripts copy the run directory to run_artifacts/ so a bench dir
    stays self-contained; fall back to the recorded run_dir for a run whose copy
    did not happen (a job killed at the wall clock, say).
    """
    copied = bench_dir / "run_artifacts"
    if (copied / "config.json").exists():
        return copied
    recorded = _read_provenance(bench_dir / "provenance.txt").get("run_dir")
    return Path(recorded) if recorded else copied


def _gpu_peak(bench_dir: Path) -> tuple[float | None, float | None]:
    """Peak GPU memory (MiB) and mean utilization (%) from the nvidia-smi samples."""
    path = bench_dir / "gpu.csv"
    if not path.exists():
        return None, None
    used: list[float] = []
    util: list[float] = []
    for line in path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            util.append(float(parts[2].rstrip(" %")))
            used.append(float(parts[4].rstrip(" MiB")))
        except ValueError:
            continue
    if not used:
        return None, None
    return max(used), sum(util) / len(util)


def _stage_table(base: Path, cand: Path, base_name: str, cand_name: str) -> None:
    b = {r["stage"]: r for r in _read_jsonl(base / "bench.jsonl")}
    c = {r["stage"]: r for r in _read_jsonl(cand / "bench.jsonl")}

    _out(f"\n{'STAGE':<10} {base_name:>22} {cand_name:>22} {'SPEEDUP':>9}")
    _out("-" * 66)

    # Known stages first, in pipeline order, then anything unexpected.
    seen = [s for s in STAGES if s in b or s in c]
    seen += sorted((set(b) | set(c)) - set(STAGES))
    for stage in seen:
        bw = b.get(stage, {}).get("wall_s")
        cw = c.get(stage, {}).get("wall_s")
        br = b.get(stage, {}).get("max_rss_kb")
        cr = c.get(stage, {}).get("max_rss_kb")
        _out(
            f"{stage:<10} "
            f"{_fmt_hms(bw) + ' / ' + _fmt_gb(br):>22} "
            f"{_fmt_hms(cw) + ' / ' + _fmt_gb(cr):>22} "
            f"{_ratio(bw, cw):>9}"
        )

    bt = sum(r.get("wall_s", 0) for r in b.values())
    ct = sum(r.get("wall_s", 0) for r in c.values())
    _out("-" * 66)
    _out(f"{'TOTAL':<10} {_fmt_hms(bt):>22} {_fmt_hms(ct):>22} {_ratio(bt, ct):>9}")

    failed = [s for s, r in {**b, **c}.items() if r.get("exit", 0) != 0]
    if failed:
        _out(f"\n  !! non-zero exit in: {', '.join(sorted(set(failed)))}")

    bg, bu = _gpu_peak(base)
    cg, cu = _gpu_peak(cand)
    if bg or cg:
        gb = f"{bg:.0f} MiB" if bg else "--"
        gc = f"{cg:.0f} MiB" if cg else "--"
        ub = f"{bu:.0f}%" if bu is not None else "--"
        uc = f"{cu:.0f}%" if cu is not None else "--"
        _out(f"\n{'GPU peak':<10} {gb:>22} {gc:>22}")
        _out(f"{'GPU mean':<10} {ub:>22} {uc:>22}")


def _metric_table(base: Path, cand: Path, fname: str, label: str) -> None:
    b = _read_json(_artifact_dir(base) / fname)
    c = _read_json(_artifact_dir(cand) / fname)
    if not b and not c:
        return

    _out(f"\n=== {label} ({fname}) ===")
    keys = sorted(set(b) | set(c))
    for family in FAMILIES:
        field = f"{family}_after"
        rows = [
            (k, b.get(k, {}).get(field), c.get(k, {}).get(field))
            for k in keys
            if field in b.get(k, {}) or field in c.get(k, {})
        ]
        if not rows:
            continue
        _out(f"\n  {family}_after")
        for key, bv, cv in rows:
            delta = "--"
            if bv not in (None, 0) and cv is not None:
                delta = f"{(cv - bv) / abs(bv) * 100:+.1f}%"
            bs = f"{bv:.6g}" if bv is not None else "--"
            cs = f"{cv:.6g}" if cv is not None else "--"
            _out(f"    {key:<22} {bs:>14} {cs:>14} {delta:>9}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="bench dir of the reference arm")
    parser.add_argument("candidate", type=Path, help="bench dir of the arm under test")
    args = parser.parse_args()

    for d in (args.baseline, args.candidate):
        if not d.is_dir():
            _err(f"not a directory: {d}")
            return 2

    bp = _read_provenance(args.baseline / "provenance.txt")
    cp = _read_provenance(args.candidate / "provenance.txt")
    base_name = bp.get("arm", args.baseline.name)
    cand_name = cp.get("arm", args.candidate.name)

    _out("=" * 78)
    _out(f"{'':<12}{'BASELINE':>32}{'CANDIDATE':>34}")
    fields = ("arm", "branch", "commit", "config", "n_samples", "seed", "data_seed")
    for field in fields:
        bv, cv = bp.get(field, "--"), cp.get(field, "--")
        _out(f"{field:<12}{bv[:32]:>32}{cv[:34]:>34}")
    _out("=" * 78)

    # A comparison across different data is not a comparison. Say so loudly
    # rather than printing a table that looks meaningful.
    for field in ("config", "n_samples", "data_seed"):
        if bp.get(field) and cp.get(field) and bp[field] != cp[field]:
            _out(f"\n  !! {field} differs between arms -- results are NOT comparable")

    _stage_table(args.baseline, args.candidate, base_name, cand_name)
    _metric_table(args.baseline, args.candidate, "metrics.json", "RAN")
    _metric_table(args.baseline, args.candidate, "metrics_omnifold.json", "OmniFold")
    _metric_table(args.baseline, args.candidate, "metrics_ibu.json", "IBU")
    _out()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
