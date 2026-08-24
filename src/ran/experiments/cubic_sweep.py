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

from ..baselines import unfold_variable
from ..data import RANDataset
from ..rantypes import (
    DEFAULT_PURITY_THRESHOLD,
    EVENT_DTYPE,
    Events,
    Populations,
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..baselines import VariableUnfolding
    from ..rantypes import DatasetSplits, EventArray, VariableOutcome

logger: Logger = logging.getLogger(name=__name__)
mpl.use(backend="Agg")

# Only s varies across the sweep, so both methods' hyperparameters are pinned
# here at what their CLI commands use rather than exposed as sweep flags.
_HIDDEN_UNITS: int = 64
_N_LAYERS: int = 2
_PATIENCE: int = 5
_IBU_ITERATIONS: int = 10


def response(s: np.single, *zs: EventArray) -> tuple[EventArray, ...]:
    """Deterministic non-linear detector response r(s, z) = z + s * z**3."""
    # NumPy-stub loses specific float precision. There is no type promotion at runtime
    return tuple(z + s * z**3.0 for z in zs)  # pyrefly: ignore[bad-return]


def _column(a: EventArray, /) -> EventArray:
    """Reshape a flat sample into the (n, 1) column the models expect."""
    return a.reshape(-1, 1)


def make_particles(n_samples: int, seed: int = 42) -> tuple[EventArray, EventArray]:
    """Draw the fixed particle-level samples the sweep distorts.

    `Generator.normal` produces float64 whatever the caller wants, so this is
    the boundary where the sweep's own data enters the pipeline's precision.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    z_truth = rng.normal(loc=0.0, scale=1.0, size=n_samples).astype(EVENT_DTYPE)
    z_gen = rng.normal(loc=-1.0, scale=1.0, size=n_samples).astype(EVENT_DTYPE)
    return z_truth, z_gen


def unfolded_wasserstein(
    z_truth: EventArray,
    z_gen: EventArray,
    weights: EventArray,
) -> np.double:
    """Wasserstein distance between z_truth and the weighted z_gen distribution."""
    return wasserstein_distance(
        z_truth.ravel(),
        z_gen.ravel(),
        v_weights=weights.ravel(),
    )


def _sweep_point(
    s_index: int, n_points: int, n_samples: int, seed: int
) -> tuple[np.single, Populations]:
    s: np.single = np.linspace(start=0.0, stop=20.0, num=n_points, dtype=EVENT_DTYPE)[
        s_index
    ]
    z_truth, z_gen = make_particles(n_samples, seed=seed)
    x_data, x_sim = response(s, z_truth, z_gen)
    return s, Populations(
        mc=Events(z=_column(z_gen), x=_column(x_sim)),
        data=_column(x_data),
        truth=_column(z_truth),
    )


def _write_point(sweep_dir: Path, out: dict) -> dict:
    """Write one point's record, coercing numpy scalars to builtins first.

    `np.float64` subclasses `float`, so `json` accepted it silently while the
    pipeline was float64. `np.float32` does not, and would raise here instead.
    Coercing on the way out keeps the record independent of the pinned dtype.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        k: v.item() if isinstance(v, np.generic) else v for k, v in out.items()
    }
    (sweep_dir / f"point_{record['s_index']:02d}.json").write_text(
        data=json.dumps(obj=record, indent=2)
    )
    return record


def _finite(w: EventArray) -> EventArray:
    """Zero out non-finite weights (e.g. a saturated classifier at large s)."""
    return np.where(np.isfinite(w), w, 0.0)


def _ibu_point(pops: Populations) -> tuple[np.double, VariableOutcome]:
    """Unfold one sweep point with IBU and score it the way RAN is scored.

    Both arms now run at the pipeline's single precision, so there is no cast
    here and nothing to keep in step: the same array reaches both unfolders.

    The fit uses `mc.z`, `mc.x` and `data` --- what a real measurement has ---
    and is applied to the same `mc.z` that RAN's weights are applied to.
    `truth` reaches neither method, and appears only in the score.
    """
    unfolding: VariableUnfolding = unfold_variable(
        variable_name="z",
        mc_gen=pops.mc.z[:, 0],
        mc_sim=pops.mc.x[:, 0],
        observed=pops.data[:, 0],
        n_iterations=_IBU_ITERATIONS,
        purity_threshold=DEFAULT_PURITY_THRESHOLD,
    )
    weights: EventArray = _finite(unfolding.weights_for(gen=pops.mc.z[:, 0]))
    return (
        unfolded_wasserstein(
            z_truth=pops.require_truth(), z_gen=pops.mc.z, weights=weights
        ),
        unfolding.outcome,
    )


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
    # Deferred so that `ran sweep collect`, which only reads JSON and plots,
    # does not pay for importing keras and jax.
    from ..train import train

    if TYPE_CHECKING:
        from ..train import TrainResult

    s, pops = _sweep_point(s_index, n_points, n_samples, seed)

    splits: DatasetSplits = RANDataset(
        batch_size=batch_size, seed=seed
    ).splits_from_data(pops.interleave())
    result: TrainResult = train(
        splits,
        dim=1,
        hidden_units=_HIDDEN_UNITS,
        n_layers=_N_LAYERS,
        patience=_PATIENCE,
        n_epochs=ran_epochs,
        seed=init_seed,
    )

    raw: EventArray = np.asarray(a=result.g(pops.mc.z)).ravel()
    w_ran: EventArray = _finite(w=raw * len(raw) / raw.sum())

    ran_wd: np.double = unfolded_wasserstein(
        z_truth=pops.require_truth(), z_gen=pops.mc.z, weights=w_ran
    )
    # IBU costs seconds next to training, so it runs here on the very same
    # populations rather than in a second pass that would have to re-derive them.
    ibu_wd, ibu_outcome = _ibu_point(pops)

    logger.info(
        "s=%.4f  RAN=%.6f  IBU=%.6f  (init seed %d, %d IBU bins)",
        s,
        ran_wd,
        ibu_wd,
        result.seed,
        ibu_outcome.n_bins,
    )
    return _write_point(
        sweep_dir,
        out={
            "s_index": s_index,
            "s": s,
            "ran_wd": ran_wd,
            "ibu_wd": ibu_wd,
            "ibu_n_bins": ibu_outcome.n_bins,
            "ibu_status": ibu_outcome.status,
            "seed": seed,
            "init_seed": result.seed,
        },
    )


def _complete_points(sweep_dir: Path, n_points: int) -> list[dict[str, Any]]:
    """Read the per-point files, keeping only the points that finished.

    Raises if nothing is complete; warns (but proceeds) when only some points
    are, so a partly-failed sweep still produces a plot of what did land.
    """
    records: dict[int, dict[str, Any]] = {}
    for f in sorted(sweep_dir.glob(pattern="point_*.json")):
        rec: dict = json.loads(s=f.read_text())
        records.setdefault(rec["s_index"], {}).update(rec)

    # Both methods are written in one pass, so a point either carries both or
    # is not there at all; requiring both is a guard against a truncated write.
    complete: list[dict[str, Any]] = sorted(
        (r for r in records.values() if "ran_wd" in r and "ibu_wd" in r),
        key=operator.itemgetter("s"),
    )
    if not complete:
        raise FileNotFoundError(f"No sweep point in {sweep_dir} has point_*.json")

    present: set[Any] = {r["s_index"] for r in complete}
    missing: list[int] = sorted(set(range(n_points)) - present)
    if missing:
        logger.warning("missing s_index values (failed/incomplete tasks): %s", missing)
    return complete


def _plot_sweep(
    sweep_dir: Path,
    s: NDArray[np.double],
    ran: NDArray[np.double],
    ibu: NDArray[np.double],
) -> None:
    """Wasserstein-vs-distortion curve for both methods."""
    figure = Figure(figsize=(7, 5))
    figure.canvas = FigureCanvasPdf(figure)
    ax: Axes = figure.subplots()
    ax.plot(s, ran, "o-", label="RAN")
    ax.plot(s, ibu, "s--", label="IBU")
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
    ibu: NDArray[np.double] = np.array(object=[r["ibu_wd"] for r in complete])
    np.savez(file=sweep_dir / "results.npz", s=s, ran=ran, ibu=ibu)
    _plot_sweep(sweep_dir, s, ran, ibu)

    skipped: list[int] = [
        r["s_index"] for r in complete if r.get("ibu_status") == "skipped"
    ]
    if skipped:
        # Not a failure: past some distortion the purity binning yields fewer
        # than two bins, IBU returns unit weights, and its curve flattens onto
        # the un-unfolded distance. Worth saying out loud so the plot is read
        # as "IBU gave up here" rather than "IBU did no worse here".
        logger.warning("IBU found too few purity bins at s_index %s", skipped)
    logger.info(
        "Wrote %s and %s (%d points)",
        sweep_dir / "results.npz",
        sweep_dir / "wasserstein_vs_s.pdf",
        len(complete),
    )
