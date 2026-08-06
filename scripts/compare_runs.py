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


def _gpu_sample(line: str) -> tuple[str, float, float] | None:
    """One nvidia-smi CSV row as (gpu index, utilization %, memory MiB)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6 or not parts[1].isdigit():
        return None
    try:
        return parts[1], float(parts[2].rstrip(" %")), float(parts[4].rstrip(" MiB"))
    except ValueError:
        return None


def _gpu_peak(bench_dir: Path) -> tuple[float | None, float | None]:
    """Peak memory (MiB) and mean utilization (%) of the GPU the run actually used.

    Per GPU index, then report the busiest one. A Perlmutter node exposes four
    A100s and neither implementation distributes across them, so averaging over
    every sampled row buries the one working GPU under three idle ones -- which
    reads as 0% utilization for a run that was genuinely busy.
    """
    path = bench_dir / "gpu.csv"
    if not path.exists():
        return None, None

    per_gpu: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines()[1:]:
        sample = _gpu_sample(line)
        if sample is not None:
            index, util, used = sample
            per_gpu.setdefault(index, []).append((util, used))

    if not per_gpu:
        return None, None
    busiest = max(per_gpu.values(), key=lambda rows: max(u for _, u in rows))
    return (
        max(u for _, u in busiest),
        sum(v for v, _ in busiest) / len(busiest),
    )


def _stage_order(b: dict[str, Any], c: dict[str, Any]) -> list[str]:
    """Known stages in pipeline order, then anything unexpected, alphabetically."""
    return [s for s in STAGES if s in b or s in c] + sorted(
        (set(b) | set(c)) - set(STAGES)
    )


def _stage_row(stage: str, b: dict[str, Any], c: dict[str, Any]) -> str:
    bw, cw = b.get("wall_s"), c.get("wall_s")
    lhs = f"{_fmt_hms(bw)} / {_fmt_gb(b.get('max_rss_kb'))}"
    rhs = f"{_fmt_hms(cw)} / {_fmt_gb(c.get('max_rss_kb'))}"
    return f"{stage:<10} {lhs:>22} {rhs:>22} {_ratio(bw, cw):>9}"


def _gpu_rows(base: Path, cand: Path) -> None:
    bg, bu = _gpu_peak(base)
    cg, cu = _gpu_peak(cand)
    if not (bg or cg):
        return
    peak = (f"{bg:.0f} MiB" if bg else "--", f"{cg:.0f} MiB" if cg else "--")
    mean = (
        f"{bu:.0f}%" if bu is not None else "--",
        f"{cu:.0f}%" if cu is not None else "--",
    )
    _out(f"\n{'GPU peak':<10} {peak[0]:>22} {peak[1]:>22}")
    _out(f"{'GPU mean':<10} {mean[0]:>22} {mean[1]:>22}")


def _stage_table(base: Path, cand: Path, base_name: str, cand_name: str) -> None:
    b = {r["stage"]: r for r in _read_jsonl(base / "bench.jsonl")}
    c = {r["stage"]: r for r in _read_jsonl(cand / "bench.jsonl")}

    _out(f"\n{'STAGE':<10} {base_name:>22} {cand_name:>22} {'SPEEDUP':>9}")
    _out("-" * 66)
    for stage in _stage_order(b, c):
        _out(_stage_row(stage, b.get(stage, {}), c.get(stage, {})))

    bt = sum(r.get("wall_s", 0) for r in b.values())
    ct = sum(r.get("wall_s", 0) for r in c.values())
    _out("-" * 66)
    _out(f"{'TOTAL':<10} {_fmt_hms(bt):>22} {_fmt_hms(ct):>22} {_ratio(bt, ct):>9}")

    failed = {s for s, r in {**b, **c}.items() if r.get("exit", 0) != 0}
    if failed:
        _out(f"\n  !! non-zero exit in: {', '.join(sorted(failed))}")

    _gpu_rows(base, cand)


def _metric_rows(
    keys: list[str], b: dict[str, Any], c: dict[str, Any], field: str
) -> list[tuple[str, float | None, float | None]]:
    return [
        (k, b.get(k, {}).get(field), c.get(k, {}).get(field))
        for k in keys
        if field in b.get(k, {}) or field in c.get(k, {})
    ]


def _metric_row(key: str, bv: float | None, cv: float | None) -> str:
    delta = "--"
    if bv not in (None, 0) and cv is not None:
        delta = f"{(cv - bv) / abs(bv) * 100:+.1f}%"
    bs = f"{bv:.6g}" if bv is not None else "--"
    cs = f"{cv:.6g}" if cv is not None else "--"
    return f"    {key:<22} {bs:>14} {cs:>14} {delta:>9}"


def _split_mismatch(b: dict[str, Any], c: dict[str, Any]) -> float | None:
    """Largest relative disagreement in `wasserstein_before` between two arms.

    `_before` is the distance between the unweighted data and MC test samples:
    no model touches it, so if the two arms scored the same events it is the
    same number. A large gap means they did not, and then no `_after` column
    here is a like-for-like comparison.

    This is not hypothetical. The tf arm partitions with
    `tf.data.Dataset.shuffle(seed=...)` and the jax arm with
    `np.random.default_rng(seed).permutation`; the same seed drives different
    algorithms, so the test splits overlap only as much as two independent
    draws would. Per-arm `_improvement_pct` stays meaningful -- each is
    measured against its own split -- but raw `_after` values are not.
    """
    worst: float | None = None
    for key in set(b) & set(c):
        bv = b[key].get("wasserstein_before")
        cv = c[key].get("wasserstein_before")
        if not bv or cv is None:
            continue
        rel = abs(cv - bv) / abs(bv)
        worst = rel if worst is None else max(worst, rel)
    return worst


def _metric_table(base: Path, cand: Path, fname: str, label: str) -> None:
    b = _read_json(_artifact_dir(base) / fname)
    c = _read_json(_artifact_dir(cand) / fname)
    if not b and not c:
        return

    _out(f"\n=== {label} ({fname}) ===")

    mismatch = _split_mismatch(b, c)
    if mismatch is not None and mismatch > 0.02:
        _out(
            f"  !! test splits differ: wasserstein_before disagrees by up to "
            f"{mismatch:.1%}."
        )
        _out("     The arms scored different events -- read _improvement_pct,")
        _out("     which is per-arm, not the raw _after columns below.")

    keys = sorted(set(b) | set(c))
    # `_improvement_pct` before `_after`: it is each arm's own before -> after,
    # so it survives the arms having scored different events. `_after` follows
    # for the absolute scale, carrying the caveat above when one applies.
    for family in FAMILIES:
        for suffix in ("_improvement_pct", "_after"):
            _emit_metric_block(keys, b, c, f"{family}{suffix}")


def _emit_metric_block(
    keys: list[str], b: dict[str, Any], c: dict[str, Any], field: str
) -> None:
    rows = _metric_rows(keys, b, c, field)
    if not rows:
        return
    _out(f"\n  {field}")
    for key, bv, cv in rows:
        _out(_metric_row(key, bv, cv))


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
        # Truncate to 30 in a 32-wide field so two long values (full commit
        # shas, most obviously) always keep whitespace between them.
        _out(f"{field:<12}{bv[:30]:>32}{cv[:30]:>34}")
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
