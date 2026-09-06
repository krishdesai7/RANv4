from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from typing import Final

    from numpy.typing import NDArray

CONFIDENCE: Final[float] = 0.95
ALPHA: Final[float] = 0.05
DEFAULT_MARGIN_PP: Final[float] = 0.5


def _read(path: Path) -> tuple[dict[int, float], str]:
    """Map seed -> mean_improvement for every SUMMARY line in a log."""
    by_seed: dict[int, float] = {}
    dtype: str = "?"
    for line in path.read_text().splitlines():
        if not line.startswith("SUMMARY "):
            continue
        fields: dict[str, str] = dict(
            token.split(sep="=", maxsplit=1) for token in line.split()[1:]
        )
        by_seed[int(fields["seed"])] = float(fields["mean_improvement"])
        dtype = fields.get("dtype", dtype)
    if not by_seed:
        raise SystemExit(f"{path}: no SUMMARY lines found")
    return by_seed, dtype


def _describe(values: NDArray[np.double], name: str, /) -> str:
    return (
        f"{name:>8}: n={values.size:<3} "
        f"mean={values.mean():7.4f}%  sd={values.std(ddof=1):.4f}  "
        f"min={values.min():7.4f}%  max={values.max():7.4f}%"
    )


def _paired_report(
    a: NDArray[np.double], b: NDArray[np.double], names: tuple[str, str], margin: float
) -> None:
    def out(text: str) -> None:
        _ = sys.stdout.write(text)

    diff: NDArray[np.double] = a - b
    n: int = diff.size
    ttest: Any = stats.ttest_rel(a, b)
    t_stat: float = ttest.statistic
    p_value: float = ttest.pvalue
    wilcoxon: Any = stats.wilcoxon(diff)
    p_wilcoxon: float = wilcoxon.pvalue

    stderr: float = diff.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(CONFIDENCE, n - 1, loc=diff.mean(), scale=stderr)
    wins = int((diff > 0).sum())

    out(f"\npaired on seed (n={n})\n")
    out(f"  mean difference   : {diff.mean():+.4f} pp ({names[0]} - {names[1]})\n")
    out(f"  95% CI            : [{lo:+.4f}, {hi:+.4f}] pp\n")
    out(f"  paired t-test     : t={t_stat:.3f}  p={p_value:.5f}\n")
    out(f"  Wilcoxon signed   : p={p_wilcoxon:.5f}\n")
    out(f"  {names[0]} better in : {wins}/{n} seeds\n")

    # A t-test can only ever fail to find a difference; it cannot show there
    # isn't one. TOST asks the question actually being asked -- "is the effect
    # small enough not to care?" -- against a margin stated up front.
    t_lower: np.double = (diff.mean() + margin) / stderr
    t_upper: np.double = (diff.mean() - margin) / stderr
    p_tost: float = max(
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
            f"\n--> Inconclusive: no significant difference (p={p_value:.3f}) "
            f"and no\n    demonstrated equivalence within +/-{margin:.2f}pp "
            f"(TOST p={p_tost:.3f}).\n    More seeds, or a margin you can defend "
            "as physically irrelevant.\n"
        )


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: compare_precision.py <log_a> <log_b> [equivalence_margin_pp]"
        )

    a_by_seed, a_name = _read(Path(sys.argv[1]))
    b_by_seed, b_name = _read(Path(sys.argv[2]))

    def out(text: str) -> None:
        _ = sys.stdout.write(text)

    out(_describe(np.array(object=list(a_by_seed.values())), a_name) + "\n")
    out(_describe(np.array(object=list(b_by_seed.values())), b_name) + "\n")

    shared: list[int] = sorted(set(a_by_seed) & set(b_by_seed))
    if len(shared) < 2:
        out(
            "\n--> Fewer than two shared seeds; cannot pair. Re-run both arms "
            "over the same seeds.\n"
        )
        return

    dropped: int = (len(a_by_seed) - len(shared)) + (len(b_by_seed) - len(shared))
    if dropped:
        out(f"\nnote: {dropped} unpaired run(s) ignored; only shared seeds count.\n")

    _paired_report(
        a=np.array(object=[a_by_seed[s] for s in shared]),
        b=np.array(object=[b_by_seed[s] for s in shared]),
        names=(a_name, b_name),
        margin=float(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_MARGIN_PP,
    )


if __name__ == "__main__":
    main()
