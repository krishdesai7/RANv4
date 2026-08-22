"""Compare two precision ensembles produced by `benchmarks/precision.py`.

    uv run python benchmarks/compare_precision.py f64.log f32.log

Answers one question: is the float32-vs-float64 difference larger than the
seed-to-seed spread within each arm? If the means differ by less than the
pooled standard deviation, the arms are indistinguishable at this ensemble
size and float32 costs nothing measurable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _read(path: Path) -> tuple[np.ndarray, str]:
    """Pull mean_improvement out of every SUMMARY line in a log."""
    values: list[float] = []
    dtype = "?"
    for line in path.read_text().splitlines():
        if not line.startswith("SUMMARY "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        values.append(float(fields["mean_improvement"]))
        dtype = fields.get("dtype", dtype)
    if not values:
        raise SystemExit(f"{path}: no SUMMARY lines found")
    return np.array(values), dtype


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_precision.py <log_a> <log_b>")

    a, a_name = _read(Path(sys.argv[1]))
    b, b_name = _read(Path(sys.argv[2]))

    out = sys.stdout.write
    for values, name in ((a, a_name), (b, b_name)):
        out(
            f"{name:>8}: n={values.size:<3} "
            f"mean={values.mean():7.4f}%  sd={values.std(ddof=1):.4f}  "
            f"min={values.min():7.4f}%  max={values.max():7.4f}%\n"
        )

    separation = abs(a.mean() - b.mean())
    # Pooled spread: the scale that a real effect would have to clear.
    pooled = float(
        np.sqrt(
            ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
            / (a.size + b.size - 2)
        )
    )
    out(f"\nseparation of means : {separation:.4f} percentage points\n")
    out(f"pooled seed spread  : {pooled:.4f} percentage points\n")

    if separation < pooled:
        out(
            "\n--> Indistinguishable: the dtype difference is smaller than the\n"
            "    seed-to-seed spread. float32 costs nothing measurable here.\n"
        )
    else:
        out(
            f"\n--> Separated by {separation / pooled:.2f}x the seed spread.\n"
            "    Worth a closer look before switching.\n"
        )


if __name__ == "__main__":
    main()
