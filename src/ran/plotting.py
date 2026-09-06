from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager

from .evaluate import _get_weights

if TYPE_CHECKING:
    from logging import Logger
    from typing import Final

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Polygon
    from numpy.typing import NDArray

    from .data import ArrayDataset
    from .rantypes import EventArray, Populations, RANModel, VarInfo

type AxesHist = tuple[
    NDArray[np.double],
    NDArray[np.double],
    BarContainer | Polygon | list[BarContainer | Polygon],
]

logger: Logger = logging.getLogger(name=__name__)

mpl.rcParams["font.family"] = "serif"
available_fonts: set[str] = {f.name for f in fontManager.ttflist}
if "Cochineal" in available_fonts:
    mpl.rcParams["font.serif"] = ["Cochineal"]
mpl.rcParams["font.size"] = 18
mpl.rcParams["text.usetex"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "0.85"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.alpha"] = 0.6
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["lines.markerfacecolor"] = "none"


# `weighted_mmd` is the unbiased U-statistic estimator, which is negative
# roughly half the time once the two distributions actually match (MMD^2 is 0
# when P = Q, so an unbiased estimator of it must cross zero). A converged
# run's criterion curve therefore has values around and below zero right where
# selection lands. A plain log axis silently masks non-positive values, which
# hides exactly the epochs a converged run cares about; `symlog` renders those
# linearly while keeping the log compression that makes the early, large
# epochs readable. `SELECTION_MMD_LINTHRESH` sets where that linear region
# starts -- near the estimator's resolution floor (~5e-4 at m=8192, see
# `train.MMD_SUBSAMPLE`), so the linear region roughly matches the noise band
# rather than being an arbitrary cutoff.
SELECTION_MMD_LINTHRESH: Final[float] = 5e-4


class _PanelSpec(NamedTuple):
    """Everything that varies between the panels of one figure."""

    nature: EventArray
    mc: EventArray
    bins: NDArray[np.double]
    xlabel: str
    title: str


class _LevelStyle(NamedTuple):
    """Everything that differs between the detector-level and particle-level figures."""

    level: str  # "detector" / "particle", used in axis labels
    symbol: str  # "x" / "z"
    title_prefix: str  # "Detector Level" / "Particle Level"
    nature_label: str  # legend entry for the reference sample
    mc_label: str  # legend entry for the simulated sample
    height_per_dim: float  # figure inches per dimension
    bins_span_both: bool  # default binning covers both samples, not just nature


def _collect_data(dataset: ArrayDataset) -> Populations:
    """Return the split as the four physics populations, each (n, dim)."""
    return dataset.as_arrays().partition()


def _hist_ratio_panel(
    ax: Axes,
    ax_r: Axes,
    x_nature: EventArray,
    x_mc: EventArray,
    w_ran: EventArray,
    bins: Sequence[float] | int,
    nature_label: str,
    mc_label: str,
    xlabel: str,
    title: str,
    w_ibu: EventArray | None = None,
) -> None:
    h_nature: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_nature,
            bins=bins,
            histtype="stepfilled",
            alpha=0.35,
            color="C0",
            label=nature_label,
        ),
    )
    h_mc: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_mc,
            bins=cast(typ=Sequence[float], val=h_nature[1]),
            histtype="stepfilled",
            alpha=0.35,
            color="C1",
            label=mc_label,
        ),
    )
    h_ran: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_mc,
            bins=cast(typ=Sequence[float], val=h_nature[1]),
            weights=w_ran,
            histtype="step",
            color="black",
            linestyle="-",
            linewidth=4,
            alpha=0.35,
            label="RAN",
        ),
    )

    bin_edges: NDArray[np.double] = h_nature[1]
    centres: NDArray[np.double] = (bin_edges[:-1] + bin_edges[1:]) / 2
    safe: NDArray[np.bool] = h_nature[0] > 0
    ratio_mc: NDArray[np.double] = np.full_like(
        a=h_nature[0],
        fill_value=np.nan,
        dtype=np.double,
    )
    ratio_ran: NDArray[np.double] = np.full_like(
        a=h_ran[0],
        fill_value=np.nan,
        dtype=np.double,
    )
    ratio_mc[safe] = h_mc[0][safe] / h_nature[0][safe]
    ratio_ran[safe] = h_ran[0][safe] / h_nature[0][safe]

    _ = ax_r.plot(
        centres,
        ratio_mc,
        color="C1",
        marker="d",
        linestyle="--",
        alpha=0.35,
    )
    _ = ax_r.plot(
        centres,
        ratio_ran,
        color="black",
        marker="o",
        linestyle="--",
        alpha=0.35,
    )

    if w_ibu is not None:
        h_ibu: AxesHist = cast(
            typ=AxesHist,
            val=ax.hist(
                x_mc,
                bins=cast(typ=Sequence[float], val=h_nature[1]),
                weights=w_ibu,
                histtype="step",
                color="green",
                linestyle=":",
                linewidth=4,
                alpha=0.35,
                label="IBU",
            ),
        )
        _ = ax.set_ylabel(ylabel="Events")
        _ = ax.legend()
        _ = ax.set_title(label=title)
        ratio_ibu: NDArray[np.double] = np.full_like(
            a=h_ibu[0], fill_value=np.nan, dtype=np.double
        )
        ratio_ibu[safe] = h_ibu[0][safe] / h_nature[0][safe]
        _ = ax_r.plot(
            centres,
            ratio_ibu,
            color="green",
            marker="s",
            linestyle="--",
            alpha=0.75,
        )
    _ = ax_r.axhline(y=1, color="gray", linewidth=0.5, alpha=0.75)
    width: float = 0.5
    _ = ax_r.set_ylim(bottom=1 - width, top=1 + width)
    _ = ax_r.set_ylabel(ylabel=f"Ratio to\n{nature_label}")
    _ = ax_r.set_xlabel(xlabel)


def _save_fig(figure: Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fname=save_path)
    logger.info("Saved %s", save_path)


_DETECTOR = _LevelStyle(
    level="detector",
    symbol="x",
    title_prefix="Detector Level",
    nature_label="Data",
    mc_label="Sim",
    height_per_dim=6,
    bins_span_both=False,
)
_PARTICLE = _LevelStyle(
    level="particle",
    symbol="z",
    title_prefix="Particle Level",
    nature_label="Truth",
    mc_label="Gen.",
    height_per_dim=6,
    bins_span_both=True,
)


def _panel_spec(
    i: int,
    dim: int,
    nature: EventArray,
    mc: EventArray,
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
) -> _PanelSpec:
    """Decide what dimension `i` shows: the arrays, binning, and labels."""
    if var_info is not None:
        cfg: VarInfo = var_info[i]
        mu: float = cfg["mu"]
        sigma: float = cfg["sigma"]
        return _PanelSpec(
            nature=nature[:, i] * sigma + mu,
            mc=mc[:, i] * sigma + mu,
            bins=np.linspace(start=cfg["xlim"][0], stop=cfg["xlim"][1], num=21),
            xlabel=cfg["symbol"],
            title=f"{cfg['xlabel']} ({style.level} level)",
        )

    nature_i: EventArray = nature[:, i]
    mc_i: EventArray = mc[:, i]
    lo: np.single = (
        min(nature_i.min(), mc_i.min()) if style.bins_span_both else nature_i.min()
    )
    hi: np.single = (
        max(nature_i.max(), mc_i.max()) if style.bins_span_both else nature_i.max()
    )
    return _PanelSpec(
        nature=nature_i,
        mc=mc_i,
        bins=np.linspace(start=lo, stop=hi, num=51),
        xlabel=(
            f"${style.symbol}_{{{i}}}$ ({style.level} level)"
            if dim > 1
            else f"{style.symbol} ({style.level} level)"
        ),
        title=(f"{style.title_prefix} — Dim {i}" if dim > 1 else style.title_prefix),
    )


def _plot_level(
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    style: _LevelStyle,
    save_path: str | Path,
    var_info: list[VarInfo] | None,
    ibu_weights: list[EventArray] | None,
) -> None:
    """Draw one stacked hist+ratio panel per dimension and save the figure."""
    dim: int = nature.shape[1]
    height: float = style.height_per_dim * dim
    figure = Figure(figsize=(8, height))
    figure.canvas = FigureCanvasPdf(figure)
    edge_margin: float = 0.5 / height
    outer_grid: GridSpec = figure.add_gridspec(
        nrows=dim,
        ncols=1,
        hspace=0.35,
        bottom=edge_margin,
        top=1 - edge_margin,
    )
    for i in range(dim):
        inner_grid: GridSpecFromSubplotSpec = outer_grid[i].subgridspec(
            nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0
        )
        ax: Axes = figure.add_subplot(inner_grid[0])
        ax_r: Axes = figure.add_subplot(inner_grid[1], sharex=ax)
        ax.tick_params(labelbottom=False)

        panel: _PanelSpec = _panel_spec(i, dim, nature, mc, var_info, style)
        _hist_ratio_panel(
            ax,
            ax_r,
            x_nature=panel.nature,
            x_mc=panel.mc,
            w_ran=w,
            bins=panel.bins.tolist(),
            nature_label=style.nature_label,
            mc_label=style.mc_label,
            xlabel=panel.xlabel,
            title=panel.title,
            w_ibu=ibu_weights[i] if ibu_weights is not None else None,
        )
    _save_fig(figure, save_path=Path(save_path))


def plot_detector_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: Path = Path("plots/detector_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
) -> None:
    test: Populations = _collect_data(test_dataset)

    _plot_level(
        nature=test.data,
        mc=test.mc.x,
        w=_get_weights(g, z_gen=test.mc.z),
        style=_DETECTOR,
        save_path=save_path,
        var_info=var_info,
        ibu_weights=ibu_weights,
    )


def plot_particle_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
) -> None:
    test: Populations = _collect_data(test_dataset)

    _plot_level(
        nature=test.require_truth(),
        mc=test.mc.z,
        w=_get_weights(g, z_gen=test.mc.z),
        style=_PARTICLE,
        save_path=save_path,
        var_info=var_info,
        ibu_weights=ibu_weights,
    )


def plot_levels(
    test_dataset: ArrayDataset,
    g: RANModel,
    detector_path: Path = Path("plots/detector_level.pdf"),
    particle_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
) -> None:
    """Draw both physics levels from one partition and generator evaluation."""
    test: Populations = _collect_data(test_dataset)
    weights: EventArray = _get_weights(g, z_gen=test.mc.z)
    _plot_level(
        nature=test.data,
        mc=test.mc.x,
        w=weights,
        style=_DETECTOR,
        save_path=detector_path,
        var_info=var_info,
        ibu_weights=ibu_weights,
    )
    _plot_level(
        nature=test.require_truth(),
        mc=test.mc.z,
        w=weights,
        style=_PARTICLE,
        save_path=particle_path,
        var_info=var_info,
        ibu_weights=ibu_weights,
    )


def plot_losses(
    history: dict[str, list[float]],
    save_path: Path = Path("plots/losses.pdf"),
) -> None:
    epochs: NDArray[np.uintc] = np.arange(len(history["train_d"]), dtype=np.uintc)

    figure: Figure = Figure(figsize=(8, 5))
    figure.canvas = FigureCanvasPdf(figure)
    ax: Axes = figure.add_subplot(111)
    train_d: NDArray[np.double] = np.array(
        object=history["train_d"],
        dtype=np.double,
    )
    val_d: NDArray[np.double] = np.array(object=history["val_d"], dtype=np.double)
    train_g: NDArray[np.double] = np.array(object=history["train_g"], dtype=np.double)
    _ = ax.plot(epochs, train_d, label="Train D", color="C0", ls=":", lw=1)
    _ = ax.plot(epochs, train_g, label="Train G", color="C1", ls=":", lw=1)
    # One validation curve, because there is one validation number: `eval_step`
    # scores both networks with a single weighted BCE, so a "Val G" line would
    # be this one drawn twice. Older runs carry a `val_g` key holding exactly
    # that copy --- it is deliberately not read.
    _ = ax.plot(epochs, val_d, label="Val D", color="C0", ls="--", lw=3, alpha=0.5)
    _ = ax.axhline(
        y=np.log(2),
        color="gray",
        linestyle="-",
        linewidth=2,
        zorder=10,
        label=r"$\log(2)$",
    )
    _ = ax.set_xlabel(xlabel="Epoch")
    _ = ax.set_ylabel(ylabel="WeightedBCE")
    _ = ax.set_title(label="Training History")
    _ = ax.legend()

    figure.tight_layout()
    figure.savefig(fname=save_path, bbox_inches="tight")
    logger.info("Saved %s", save_path)


def plot_selection(
    history: dict[str, list[float]],
    best_epoch: int,
    save_path: Path = Path("plots/selection.pdf"),
) -> None:
    """The two MMD curves and the epoch selection landed on.

    Detector-level MMD is the criterion; particle-level is the diagnostic.
    Where they diverge -- detector still falling while particle turns up -- is
    the ill-posedness made visible, and it is the plot that answers whether
    truth-free selection costs anything. The particle curve is absent for a
    real measurement, which has no truth to score against, so it is optional.

    ESS shares the figure because the adversarial objective is linear in the
    weights and therefore maximized at a simplex vertex: a falling MMD bought
    by a collapsing effective sample size is not an improvement.
    """
    epochs: NDArray[np.uintc] = np.arange(len(history["val_mmd"]), dtype=np.uintc)

    figure: Figure = Figure(figsize=(8, 5))
    figure.canvas = FigureCanvasPdf(figure)
    ax: Axes = figure.add_subplot(111)

    _ = ax.plot(
        epochs,
        np.array(history["val_mmd"], dtype=np.double),
        label="Detector MMD$^2$ (criterion)",
        color="C0",
        lw=2,
    )
    if "val_mmd_particle" in history:
        _ = ax.plot(
            epochs,
            np.array(history["val_mmd_particle"], dtype=np.double),
            label="Particle MMD$^2$ (diagnostic)",
            color="C3",
            ls="--",
            lw=2,
        )
    if best_epoch >= 0:
        _ = ax.axvline(
            best_epoch,
            color="k",
            ls=":",
            lw=1,
            label=f"selected (epoch {best_epoch + 1})",
        )
    ax.set_yscale(value="symlog", linthresh=SELECTION_MMD_LINTHRESH)
    _ = ax.set_xlabel(xlabel="Epoch")
    _ = ax.set_ylabel(ylabel=r"MMD$^2$")

    ess: Axes = ax.twinx()
    _ = ess.plot(
        epochs,
        np.array(history["val_ess"], dtype=np.double),
        color="C7",
        lw=1,
        alpha=0.6,
    )
    _ = ess.set_ylabel(ylabel="Effective sample size", color="C7")
    ess.tick_params(axis="y", labelcolor="C7")

    _ = ax.legend(loc="best")
    figure.tight_layout()
    figure.savefig(fname=save_path)
    logger.info("Saved %s", save_path)
