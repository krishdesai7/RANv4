"""Sweep the cubic detector-distortion strength s and compare RAN vs OmniFold.

For each s in linspace(0, 20, n_points), apply a deterministic non-linear
detector response r(s, z) = z + s * z**3 to fixed particle-level samples
(z_truth ~ N(0,1), z_gen ~ N(-1,1)), unfold z_gen back toward z_truth with both
RAN and OmniFold, and record Wasserstein(z_truth, z_unfolded) for each.

RAN and OmniFold run as separate subcommands because they need different Keras
backends in different processes (RAN on JAX, OmniFold on TensorFlow -- see
ran/baselines/omnifold.py). Each writes its own per-point JSON; `collect` joins
them on s_index. Neither import happens at module scope, so importing this
module commits to neither backend.

Usage:
    uv run -m ran sweep ran --s-index 0 --sweep-dir ...
    uv run -m ran sweep omnifold --s-index 0 --sweep-dir ...
    uv run -m ran sweep collect --sweep-dir ...
"""

import json
import logging
import operator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from scipy.stats import wasserstein_distance

from ..data import RANDataset
from ..rantypes import Events, Populations

logger = logging.getLogger(__name__)


def response(s: float, z: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Deterministic non-linear detector response r(s, z) = z + s * z**3."""
    return z + s * z**3


def _column(a: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Reshape a flat sample into the (n, 1) float64 column the models expect."""
    return a.reshape(-1, 1).astype(np.double)


def make_particles(
    n_samples: int, seed: int = 42
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Draw fixed particle-level samples: z_truth ~ N(0,1), z_gen ~ N(-1,1)."""
    rng = np.random.default_rng(seed)
    z_truth = rng.normal(0.0, 1.0, size=n_samples)
    z_gen = rng.normal(-1.0, 1.0, size=n_samples)
    return z_truth, z_gen


def unfolded_wasserstein[T: np.floating](
    z_truth: npt.NDArray[T],
    z_gen: npt.NDArray[T],
    weights: npt.NDArray[T],
) -> float:
    """Wasserstein distance between z_truth and the weighted z_gen distribution."""
    return float(
        wasserstein_distance(
            np.asarray(z_truth).ravel(),
            np.asarray(z_gen).ravel(),
            v_weights=np.asarray(weights).ravel(),
        )
    )


def _sweep_point(
    s_index: int, n_points: int, n_samples: int, seed: int
) -> tuple[float, ndarray, ndarray, ndarray, ndarray]:
    """Resolve one sweep point: its s, the fixed particles, and their response."""
    s = float(np.linspace(0.0, 20.0, n_points)[s_index])
    z_truth, z_gen = make_particles(n_samples, seed=seed)
    return s, z_truth, z_gen, response(s, z_truth), response(s, z_gen)


def _write_point(sweep_dir: Path, prefix: str, out: dict) -> dict:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f"{prefix}_{out['s_index']:02d}.json").write_text(
        json.dumps(out, indent=2)
    )
    return out


def _finite[T: np.floating](w: npt.NDArray[T]) -> npt.NDArray[T]:
    """Zero out non-finite weights (e.g. a saturated classifier at large s).

    Keeps the Wasserstein call from crashing; bad weights just carry no mass.
    """
    return np.where(np.isfinite(w), w, 0.0)


def run_ran(
    s_index: int,
    sweep_dir: str | Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    batch_size: int = 1024,
    ran_epochs: int = 100,
    init_seed: int | None = None,
) -> dict:
    """Train RAN at one sweep point and write ran_{index}.json.

    Particle samples are drawn once (fixed `seed`) so only s varies across
    points. z_unfolded is the z_gen sample reweighted by g(z_gen).

    Arguments:
        seed: Particle-sample and dataset seed. Keep fixed across the sweep.
        init_seed: Weight-initialization seed, drawn from entropy when omitted.
            Re-run a point with different values to get an ensemble at fixed s;
            the resolved value is recorded in the output JSON.
    """
    # Deferred: importing ran.train pins Keras to JAX (see module docstring).
    from ..train import train

    s, z_truth, z_gen, x_data, x_sim = _sweep_point(s_index, n_points, n_samples, seed)

    data = Populations(
        mc=Events(_column(z_gen), _column(x_sim)),
        data=_column(x_data),
        truth=_column(z_truth),
    ).interleave()
    splits = RANDataset(batch_size=batch_size, seed=seed).splits_from_data(data)
    result = train(splits, dim=1, n_epochs=ran_epochs, seed=init_seed)

    raw = np.asarray(result.g(z_gen.reshape(-1, 1).astype(np.double))).ravel()
    w_ran = _finite(raw * len(raw) / raw.sum())

    ran_wd = unfolded_wasserstein(z_truth, z_gen, w_ran)
    logger.info("s=%.4f  RAN=%.6f  (init seed %d)", s, ran_wd, result.seed)
    return _write_point(
        Path(sweep_dir),
        "ran",
        {
            "s_index": s_index,
            "s": s,
            "ran_wd": ran_wd,
            "seed": seed,
            "init_seed": result.seed,
        },
    )


def run_omnifold(
    s_index: int,
    sweep_dir: str | Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    omnifold_niter: int = 3,
    omnifold_epochs: int = 50,
    omnifold_batch_size: int = 512,
) -> dict:
    """Unfold with OmniFold at one sweep point and write omnifold_{index}.json.

    Draws the same fixed particle samples as run_ran, so the two subcommands
    compare like for like. Must run in its own process: the import below pins
    Keras to the TensorFlow backend.
    """
    from ..baselines.omnifold import omnifold_unfold

    s, z_truth, z_gen, x_data, x_sim = _sweep_point(s_index, n_points, n_samples, seed)

    # OmniFold derives validation_steps = (0.2 * NTRAIN) // batch_size from the
    # ~2*n_samples reco events; if that floors to 0 its model.fit hangs forever
    # on a repeating validation dataset. Cap the batch size at 2*n/5 so there is
    # always >= 1 validation step (no-op at the real n=500k scale).
    of_batch = max(1, min(omnifold_batch_size, 2 * n_samples // 5))
    w_of = _finite(
        omnifold_unfold(
            x_data.reshape(-1, 1),
            x_sim.reshape(-1, 1),
            z_gen.reshape(-1, 1),
            niter=omnifold_niter,
            epochs=omnifold_epochs,
            batch_size=of_batch,
            # Per-point subdirectory, not sweep_dir itself: submit_sweep.sh runs
            # every point from the same cwd, concurrently, so a shared out_dir
            # would have them truncating each other's OmniFold log.
            out_dir=Path(sweep_dir) / f"omnifold_{s_index:02d}",
        )
    )

    of_wd = unfolded_wasserstein(z_truth, z_gen, w_of)
    logger.info("s=%.4f  OmniFold=%.6f", s, of_wd)
    return _write_point(
        Path(sweep_dir),
        "omnifold",
        {"s_index": s_index, "s": s, "omnifold_wd": of_wd},
    )


def _complete_points(sweep_dir: Path, n_points: int) -> list[dict]:
    """Join the per-method point files, keeping only points both methods finished.

    Raises if nothing is complete; warns (but proceeds) when only some points
    are, so a partly-failed sweep still produces a plot of what did land.
    """
    records: dict[int, dict] = {}
    for prefix in ("ran", "omnifold"):
        for f in sorted(sweep_dir.glob(f"{prefix}_*.json")):
            rec = json.loads(f.read_text())
            records.setdefault(rec["s_index"], {}).update(rec)

    complete = sorted(
        (r for r in records.values() if "ran_wd" in r and "omnifold_wd" in r),
        key=operator.itemgetter("s"),
    )
    if not complete:
        raise FileNotFoundError(
            f"No sweep point in {sweep_dir} has both ran_*.json and omnifold_*.json"
        )

    present = {r["s_index"] for r in complete}
    missing = sorted(set(range(n_points)) - present)
    if missing:
        logger.warning("missing s_index values (failed/incomplete tasks): %s", missing)
    return complete


def _plot_sweep(
    sweep_dir: Path,
    s: npt.NDArray,
    ran: npt.NDArray,
    omnifold: npt.NDArray,
) -> None:
    """Wasserstein-vs-distortion curve for both methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, ran, "o-", label="RAN")
    ax.plot(s, omnifold, "s-", label="OmniFold")
    ax.set_xlabel(r"$s$ (cubic distortion strength)")
    ax.set_ylabel(r"Wasserstein($z_\mathrm{truth}$, $z_\mathrm{unfolded}$)")
    ax.set_title("Unfolding performance vs detector distortion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_dir / "wasserstein_vs_s.pdf")
    plt.close(fig)


def collect(sweep_dir: str | Path, n_points: int = 25) -> None:
    """Join the per-method point files into results.npz and a Wasserstein-vs-s PDF.

    A point contributes only if both methods wrote it; points where either side
    failed are reported as missing rather than silently half-plotted.
    """
    sweep_dir = Path(sweep_dir)
    complete = _complete_points(sweep_dir, n_points)

    s = np.array([r["s"] for r in complete])
    ran = np.array([r["ran_wd"] for r in complete])
    omnifold = np.array([r["omnifold_wd"] for r in complete])
    np.savez(sweep_dir / "results.npz", s=s, ran=ran, omnifold=omnifold)
    _plot_sweep(sweep_dir, s, ran, omnifold)
    logger.info(
        "Wrote %s and %s (%d points)",
        sweep_dir / "results.npz",
        sweep_dir / "wasserstein_vs_s.pdf",
        len(complete),
    )
