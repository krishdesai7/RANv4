"""Compare two precision ensembles produced by `benchmarks/precision.py`.

    uv run python benchmarks/compare_precision.py f64.log f32.log [margin_pp]

The runs are **paired**: the same `--seed` initializes the same weights in both
arms, on the same data. Pairing is most of the available statistical power
here, because seed-to-seed variation (sd ~0.4pp) is several times the effect
being looked for (~0.2pp), and it cancels in the within-pair difference.

Three mistakes earlier versions of this file made, kept here as warnings
because each one produced a confident and wrong verdict:

1. Comparing the separation of means to the *pooled standard deviation*. That
   is the spread of individual runs, not of their mean; the right scale is the
   standard error of the difference, smaller by sqrt(1/n1 + 1/n2).
2. Treating paired runs as independent samples, discarding the pairing.
3. Reporting "n needed for 80% power" computed from the *observed* effect. An
   effect estimated near p=0.05 at small n is inflated, so that figure is far
   too small. It read ~16 seeds at n=10; ten more seeds shrank the estimate and
   it would have read ~75.

The deeper lesson is in the data rather than the code: at n=10 this showed
+0.27pp with p=0.052 and float64 ahead 8/10. Ten fresh seeds showed +0.12pp,
p=0.64, float64 ahead 4/10. The first result was a fluctuation. A t-test can
only ever fail to find a difference, so the affirmative question -- "is the gap
small enough not to care?" -- is answered by TOST against a stated margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

CONFIDENCE: float = 0.95
ALPHA: float = 0.05
# How much unfolding improvement you are willing to trade, in percentage points.
# Equivalence is only ever "within a margin"; the margin is a physics judgement
# and has to be stated, so it is an argument rather than a hidden default.
DEFAULT_MARGIN_PP: float = 0.5


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


def _paired_report(
    a: np.ndarray, b: np.ndarray, names: tuple[str, str], margin: float
) -> None:
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

    # A t-test can only ever fail to find a difference; it cannot show there
    # isn't one. TOST asks the question actually being asked -- "is the effect
    # small enough not to care?" -- against a margin stated up front.
    t_lower = (diff.mean() + margin) / stderr
    t_upper = (diff.mean() - margin) / stderr
    p_tost = max(
        float(1 - stats.t.cdf(t_lower, n - 1)), float(stats.t.cdf(t_upper, n - 1))
    )
    out(f"  TOST +/-{margin:.2f}pp    : p={p_tost:.5f}\n")

    if p_value < ALPHA:
        out(
            f"\n--> Difference significant at alpha={ALPHA}. Whether "
            f"{abs(diff.mean()):.2f}pp of unfolding improvement\n    matters is a "
            "physics call, not a statistics one.\n"
        )
    elif p_tost < ALPHA:
        out(
            f"\n--> EQUIVALENT within +/-{margin:.2f}pp (TOST p={p_tost:.4f}).\n"
            f"    Not merely 'no difference found': the effect is demonstrably\n"
            f"    smaller than the margin you said you care about.\n"
        )
    else:
        out(
            f"\n--> Inconclusive: no significant difference (p={float(p_value):.3f}) "
            f"and no\n    demonstrated equivalence within +/-{margin:.2f}pp "
            f"(TOST p={p_tost:.3f}).\n    More seeds, or a margin you can defend "
            "as physically irrelevant.\n"
        )

    # Deliberately no "n needed for 80% power" from the observed effect. An
    # effect estimated near p=0.05 at small n is inflated (Type M error), so
    # that number comes out far too small -- it read ~16 at n=10 here, and ~75
    # once ten more seeds shrank the estimate.


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: compare_precision.py <log_a> <log_b> [equivalence_margin_pp]"
        )

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
        float(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_MARGIN_PP,
    )


if __name__ == "__main__":
    main()
