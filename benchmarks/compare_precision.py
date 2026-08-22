"""Compare two precision ensembles produced by `benchmarks/precision.py`.

    uv run python benchmarks/compare_precision.py f64.log f32.log

The runs are **paired**: the same `--seed` initializes the same weights in both
arms, on the same data. Pairing is most of the available statistical power
here, because seed-to-seed variation (sd ~0.4pp) is several times the effect
being looked for (~0.27pp), and it cancels in the within-pair difference.

Two things this deliberately does not do, because an earlier version did both
and reported "indistinguishable" for a p=0.05 effect:

1. Compare the separation of means to the *pooled standard deviation*. That is
   the spread of individual runs, not of their mean; the right scale is the
   standard error of the difference, smaller by sqrt(1/n1 + 1/n2).
2. Treat paired runs as independent samples, which throws the pairing away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

CONFIDENCE: float = 0.95
ALPHA: float = 0.05
# z(0.975) + z(0.80): the constant in the standard two-sided power formula.
_POWER_Z: float = 2.802


def _read(path: Path) -> tuple[dict[int, float], str]:
    """Map seed -> mean_improvement for every SUMMARY line in a log."""
    by_seed: dict[int, float] = {}
    dtype = "?"
    for line in path.read_text().splitlines():
        if not line.startswith("SUMMARY "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        by_seed[int(fields["seed"])] = float(fields["mean_improvement"])
        dtype = fields.get("dtype", dtype)
    if not by_seed:
        raise SystemExit(f"{path}: no SUMMARY lines found")
    return by_seed, dtype


def _describe(values: np.ndarray, name: str) -> str:
    return (
        f"{name:>8}: n={values.size:<3} "
        f"mean={values.mean():7.4f}%  sd={values.std(ddof=1):.4f}  "
        f"min={values.min():7.4f}%  max={values.max():7.4f}%"
    )


def _paired_report(a: np.ndarray, b: np.ndarray, names: tuple[str, str]) -> None:
    out = sys.stdout.write
    diff = a - b
    n = diff.size
    t_stat, p_value = stats.ttest_rel(a, b)
    _, p_wilcoxon = stats.wilcoxon(diff)

    stderr = float(diff.std(ddof=1) / np.sqrt(n))
    lo, hi = stats.t.interval(CONFIDENCE, n - 1, loc=diff.mean(), scale=stderr)
    wins = int((diff > 0).sum())

    out(f"\npaired on seed (n={n})\n")
    out(f"  mean difference   : {diff.mean():+.4f} pp ({names[0]} - {names[1]})\n")
    out(f"  95% CI            : [{lo:+.4f}, {hi:+.4f}] pp\n")
    out(f"  paired t-test     : t={float(t_stat):.3f}  p={float(p_value):.5f}\n")
    out(f"  Wilcoxon signed   : p={float(p_wilcoxon):.5f}\n")
    out(f"  {names[0]} better in : {wins}/{n} seeds\n")

    if p_value < ALPHA:
        out(
            f"\n--> Significant at alpha={ALPHA}. The dtype difference is real at "
            f"this ensemble size.\n    Whether {abs(diff.mean()):.2f}pp of "
            "unfolding improvement matters is a physics call, not a statistics one.\n"
        )
        return

    # Not significant is not the same as no effect -- say which one this is.
    effect = abs(diff.mean()) / float(diff.std(ddof=1))
    needed = int(np.ceil((_POWER_Z / effect) ** 2)) if effect > 0 else 0
    out(f"\n--> Not significant at alpha={ALPHA}, but this is not 'no effect':\n")
    out(f"    the CI spans {lo:+.4f} to {hi:+.4f}, and {names[0]} won {wins}/{n}.\n")
    out(f"    For 80% power at this effect size you need ~{needed} paired seeds.\n")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_precision.py <log_a> <log_b>")

    a_by_seed, a_name = _read(Path(sys.argv[1]))
    b_by_seed, b_name = _read(Path(sys.argv[2]))

    out = sys.stdout.write
    out(_describe(np.array(list(a_by_seed.values())), a_name) + "\n")
    out(_describe(np.array(list(b_by_seed.values())), b_name) + "\n")

    shared = sorted(set(a_by_seed) & set(b_by_seed))
    if len(shared) < 2:
        out(
            "\n--> Fewer than two shared seeds; cannot pair. Re-run both arms "
            "over the same seeds.\n"
        )
        return

    dropped = (len(a_by_seed) - len(shared)) + (len(b_by_seed) - len(shared))
    if dropped:
        out(f"\nnote: {dropped} unpaired run(s) ignored; only shared seeds count.\n")

    _paired_report(
        np.array([a_by_seed[s] for s in shared]),
        np.array([b_by_seed[s] for s in shared]),
        (a_name, b_name),
    )


if __name__ == "__main__":
    main()
