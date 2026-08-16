from __future__ import annotations

import json
import logging
import operator
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from scipy.stats import wasserstein_distance

from ..data import RANDataset
from ..rantypes import Events, Populations

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..rantypes import DatasetSplits

logger: Logger = logging.getLogger(name=__name__)
mpl.use(backend="Agg")


def response[T: np.floating = np.double](
    s: T, *zs: NDArray[T]
) -> tuple[NDArray[T], ...]:
    """Deterministic non-linear detector response r(s, z) = z + s * z**3."""
    # NumPy-stub loses specific float precision. There is no type promotion at runtime
    return tuple(z + s * z**3.0 for z in zs)  # pyrefly: ignore[bad-return]


def _column[T: np.floating = np.double](a: NDArray[T], /) -> NDArray[T]:
    """Reshape a flat sample into the (n, 1) column the models expect."""
    return a.reshape(-1, 1)


def make_particles(
    n_samples: int, seed: int = 42
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    rng: np.random.Generator = np.random.default_rng(seed)
    z_truth: NDArray[np.double] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    z_gen: NDArray[np.double] = rng.normal(loc=-1.0, scale=1.0, size=n_samples)
    return z_truth, z_gen


def unfolded_wasserstein[T: np.floating = np.double](
    z_truth: NDArray[T],
    z_gen: NDArray[T],
    weights: NDArray[T],
) -> np.double:
    """Wasserstein distance between z_truth and the weighted z_gen distribution."""
    return wasserstein_distance(
        z_truth.ravel(),
        z_gen.ravel(),
        v_weights=weights.ravel(),
    )


def _sweep_point(
    s_index: int, n_points: int, n_samples: int, seed: int
) -> tuple[np.double, Populations[np.double]]:
    s: np.double = np.linspace(start=0.0, stop=20.0, num=n_points)[s_index]
    z_truth, z_gen = make_particles(n_samples, seed=seed)
    x_data, x_sim = response(s, z_truth, z_gen)
    return s, Populations(
        mc=Events(z=_column(z_gen), x=_column(x_sim)),
        data=_column(x_data),
        truth=_column(z_truth),
    )


def _write_point(sweep_dir: Path, prefix: str, out: dict) -> dict:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f"{prefix}_{out['s_index']:02d}.json").write_text(
        data=json.dumps(obj=out, indent=2)
    )
    return out


def _finite[T: np.floating = np.double](w: NDArray[T]) -> NDArray[T]:
    """Zero out non-finite weights (e.g. a saturated classifier at large s)."""
    return np.where(np.isfinite(w), w, 0.0)


def run_ran(
    s_index: int,
    sweep_dir: Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    batch_size: int = 1024,
    ran_epochs: int = 100,
    init_seed: int | None = None,
) -> dict:
    # Deferred: importing ran.train pins Keras to JAX (see module docstring).
    from ..train import train

    if TYPE_CHECKING:
        from ..train import TrainResult

    s, pops = _sweep_point(s_index, n_points, n_samples, seed)

    splits: DatasetSplits[np.double] = RANDataset(
        batch_size=batch_size, seed=seed
    ).splits_from_data(pops.interleave())
    result: TrainResult = train(splits, dim=1, n_epochs=ran_epochs, seed=init_seed)

    raw: NDArray[np.double] = np.asarray(a=result.g(pops.mc.z)).ravel()
    w_ran: NDArray[np.double] = _finite(w=raw * len(raw) / raw.sum())

    ran_wd: np.double = unfolded_wasserstein(
        z_truth=pops.require_truth(), z_gen=pops.mc.z, weights=w_ran
    )
    logger.info("s=%.4f  RAN=%.6f  (init seed %d)", s, ran_wd, result.seed)
    return _write_point(
        sweep_dir,
        prefix="ran",
        out={
            "s_index": s_index,
            "s": s,
            "ran_wd": ran_wd,
            "seed": seed,
            "init_seed": result.seed,
        },
    )


def run_omnifold(
    s_index: int,
    sweep_dir: Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    omnifold_niter: int = 3,
    omnifold_epochs: int = 50,
    omnifold_batch_size: int = 512,
) -> dict:
    from ..baselines.omnifold import omnifold_unfold

    s, pops = _sweep_point(s_index, n_points, n_samples, seed)

    # OmniFold derives validation_steps=0.2*NTRAIN//batch_size from reco events; if that
    # floors to 0 model.fit hangs forever. Cap batch size at 2*n/5 so there is always
    # >= 1 validation step.
    of_batch: int = max(1, min(omnifold_batch_size, 2 * n_samples // 5))
    w_of: NDArray[np.single] = _finite(
        omnifold_unfold(
            x_data=pops.data,
            x_sim=pops.mc.x,
            z_gen=pops.mc.z,
            niter=omnifold_niter,
            epochs=omnifold_epochs,
            batch_size=of_batch,
            # Per-point subdirectory, not sweep_dir: submit_sweep.sh runs every point
            # from cwd concurrently, so shared out_dir would truncate OmniFold logs.
            out_dir=sweep_dir / f"omnifold_{s_index:02d}",
        )
    )

    of_wd: np.double = unfolded_wasserstein(
        z_truth=pops.require_truth(), z_gen=pops.mc.z, weights=w_of
    )
    logger.info("s=%.4f  OmniFold=%.6f", s, of_wd)
    return _write_point(
        sweep_dir,
        prefix="omnifold",
        out={"s_index": s_index, "s": s, "omnifold_wd": of_wd},
    )


def _complete_points(sweep_dir: Path, n_points: int) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for prefix in ("ran", "omnifold"):
        for f in sorted(sweep_dir.glob(pattern=f"{prefix}_*.json")):
            rec: dict = json.loads(s=f.read_text())
            records.setdefault(rec["s_index"], {}).update(rec)

    complete: list[dict[str, Any]] = sorted(
        (r for r in records.values() if "ran_wd" in r and "omnifold_wd" in r),
        key=operator.itemgetter("s"),
    )
    if not complete:
        raise FileNotFoundError(
            f"No sweep point in {sweep_dir} has both ran_*.json and omnifold_*.json"
        )

    present: set[Any] = {r["s_index"] for r in complete}
    missing: list[int] = sorted(set(range(n_points)) - present)
    if missing:
        logger.warning("missing s_index values (failed/incomplete tasks): %s", missing)
    return complete


def _plot_sweep(
    sweep_dir: Path,
    s: NDArray[np.double],
    ran: NDArray[np.double],
    omnifold: NDArray[np.double],
) -> None:
    """Wasserstein-vs-distortion curve for both methods."""
    figure = Figure(figsize=(7, 5))
    figure.canvas = FigureCanvasPdf(figure)
    ax: Axes = figure.subplots()
    ax.plot(s, ran, "o-", label="RAN")
    ax.plot(s, omnifold, "s-", label="OmniFold")
    ax.set_xlabel(xlabel=r"$s$ (cubic distortion strength)")
    ax.set_ylabel(ylabel=r"Wasserstein($z_\mathrm{truth}$, $z_\mathrm{unfolded}$)")
    ax.set_title(label="Unfolding performance vs detector distortion")
    ax.legend()
    figure.tight_layout()
    figure.savefig(fname=sweep_dir / "wasserstein_vs_s.pdf")


def collect(sweep_dir: Path, n_points: int = 25) -> None:
    complete: list[dict[str, Any]] = _complete_points(sweep_dir, n_points)

    s: NDArray[np.double] = np.array(object=[r["s"] for r in complete])
    ran: NDArray[np.double] = np.array(object=[r["ran_wd"] for r in complete])
    omnifold: NDArray[np.double] = np.array(object=[r["omnifold_wd"] for r in complete])
    np.savez(file=sweep_dir / "results.npz", s=s, ran=ran, omnifold=omnifold)
    _plot_sweep(sweep_dir, s, ran, omnifold)
    logger.info(
        "Wrote %s and %s (%d points)",
        sweep_dir / "results.npz",
        sweep_dir / "wasserstein_vs_s.pdf",
        len(complete),
    )
