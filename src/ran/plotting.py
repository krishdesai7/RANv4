from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MaxNLocator

from .evaluate import _get_weights
from .rantypes import display_order

if TYPE_CHECKING:
    from logging import Logger
    from typing import Final

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
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


# One place for the figure's visual hierarchy, rather than seven literals
# scattered through `_hist_ratio_panel`. RAN's step line used to be black at
# alpha 0.35 while IBU's ratio line was at 0.75 -- the baseline drawn twice as
# prominently as the method being showcased, on the same panel.
COLOR_NATURE: Final[str] = "C0"  # Data / Truth
COLOR_MC: Final[str] = "C1"  # Sim / Gen
COLOR_IBU: Final[str] = "green"
COLOR_RAN: Final[str] = "#6A3D9A"  # deep violet; greyscales to a dark mid-tone

ALPHA_FILL: Final[float] = 0.35  # the two filled background histograms
ALPHA_IBU: Final[float] = 0.75
ALPHA_RAN: Final[float] = 0.90


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
            alpha=ALPHA_FILL,
            color=COLOR_NATURE,
            label=nature_label,
        ),
    )
    h_mc: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_mc,
            bins=cast(typ=Sequence[float], val=h_nature[1]),
            histtype="stepfilled",
            alpha=ALPHA_FILL,
            color=COLOR_MC,
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
            color=COLOR_RAN,
            linestyle="-",
            linewidth=4,
            alpha=ALPHA_RAN,
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
        color=COLOR_MC,
        marker="d",
        linestyle="--",
        alpha=ALPHA_FILL,
    )
    _ = ax_r.plot(
        centres,
        ratio_ran,
        color=COLOR_RAN,
        marker="o",
        linestyle="--",
        alpha=ALPHA_RAN,
    )

    if w_ibu is not None:
        h_ibu: AxesHist = cast(
            typ=AxesHist,
            val=ax.hist(
                x_mc,
                bins=cast(typ=Sequence[float], val=h_nature[1]),
                weights=w_ibu,
                histtype="step",
                color=COLOR_IBU,
                linestyle=":",
                linewidth=4,
                alpha=ALPHA_IBU,
                label="IBU",
            ),
        )
        ratio_ibu: NDArray[np.double] = np.full_like(
            a=h_ibu[0], fill_value=np.nan, dtype=np.double
        )
        ratio_ibu[safe] = h_ibu[0][safe] / h_nature[0][safe]
        _ = ax_r.plot(
            centres,
            ratio_ibu,
            color=COLOR_IBU,
            marker="s",
            linestyle="--",
            alpha=ALPHA_IBU,
        )
    # Every panel gets a label, a title and a legend, not only one drawn
    # against an IBU baseline -- `ibu_weights.npz` does not exist on the
    # default `ran train` path, and until it does every panel was unlabelled,
    # untitled and legend-less. `ax.legend()` runs once here, after the IBU
    # branch, so it picks up the "IBU" handle when that branch ran and omits
    # it otherwise.
    _ = ax.set_ylabel(ylabel="Events")
    _ = ax.set_title(label=title)
    _ = ax.legend()
    _ = ax_r.axhline(y=1, color="gray", linewidth=0.5, alpha=0.75)
    width: float = 0.5
    _ = ax_r.set_ylim(bottom=1 - width, top=1 + width)
    _ = ax_r.set_ylabel(ylabel=f"Ratio to\n{nature_label}")
    # The main panel's bottom tick and the ratio panel's top tick land at the
    # same height where the two axes meet and overprint each other. `prune`
    # drops the lowest label only when it sits at the axis edge, which is
    # exactly the collision and nothing else.
    ax.yaxis.set_major_locator(locator=MaxNLocator(prune="lower"))
    _ = ax_r.set_xlabel(xlabel)


def _save_fig(figure: Figure, save_path: Path) -> None:
    """Save `figure`, trimmed to its rendered contents.

    Without `bbox_inches="tight"` the y-labels are clipped by the page edge.
    `plot_losses` always passed it and never clipped; `plot_selection` and
    `_plot_level` did not and did -- wide tick labels (e.g. five-digit event
    counts) push the y-label further left than `_plot_level`'s fixed
    `GridSpec` margins reserve for it, so a real run's `detector_level.pdf`
    and `particle_level.pdf` clip even though a narrower synthetic figure
    does not. All three now go through this one save path instead of calling
    `figure.savefig` themselves.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fname=save_path, bbox_inches="tight")
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
            # No "(detector/particle level)" suffix here: at a 4-inch panel
            # width in the grid `_plot_level` lays out, repeating it on all
            # twelve panels made adjacent titles overlap (worst case 109.5px).
            # `_plot_level`'s `figure.suptitle` states the level once for the
            # whole figure instead.
            title=cfg["xlabel"],
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
        # `_plot_level`'s `figure.suptitle` now states `style.title_prefix`
        # once for the whole figure, so a panel title repeating it here --
        # even the single-dimension case's old bare `style.title_prefix` --
        # would duplicate it. `dim > 1` still names which dimension a panel
        # is; `dim == 1` has nothing left to say.
        title=(f"Dim {i}" if dim > 1 else ""),
    )


def _draw_panel(
    figure: Figure,
    cell: SubplotSpec,
    i: int,
    dim: int,
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
    ibu_weights: list[EventArray] | None,
) -> None:
    """Draw dimension `i`'s stacked hist+ratio panel into `cell`."""
    inner_grid: GridSpecFromSubplotSpec = cell.subgridspec(
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


def _plot_level(
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    style: _LevelStyle,
    save_path: str | Path,
    var_info: list[VarInfo] | None,
    ibu_weights: list[EventArray] | None,
    variables: tuple[str, ...] | None = None,
) -> None:
    """Draw one stacked hist+ratio panel per dimension, laid out as a grid.

    Panels are at most 3 to a row, and ordered by `display_order` on
    `variables` (or the `dim_i` identity for a non-jet run) rather than by
    raw column index, so a 12-observable jet run reads as a 4x3 grid in
    physics order instead of a 1x12 column.
    """
    dim: int = nature.shape[1]
    ncols: int = min(3, dim)
    nrows: int = math.ceil(dim / ncols)
    figure = Figure(figsize=(4.0 * ncols, style.height_per_dim * nrows))
    figure.canvas = FigureCanvasPdf(figure)
    # Absolute margins in inches do not survive a figure whose height now
    # varies with `nrows`; `tight_layout` at the end replaces them. Row/column
    # spacing goes through `tight_layout`'s own `h_pad` below rather than an
    # `hspace=` here: passing `hspace` marks this `GridSpec` as "locally
    # modified" (`GridSpec.locally_modified_subplot_params`), which makes
    # `tight_layout` treat every nested Axes as unrecognized and silently fall
    # back to Matplotlib's default (too-small) margins instead of computed
    # ones -- visible as axis labels rendered off the left edge of the page.
    outer_grid: GridSpec = figure.add_gridspec(nrows=nrows, ncols=ncols)
    # States the level once for the whole figure instead of on every panel
    # title -- see `_panel_spec`. `rect` reserves a slice of the figure height
    # above `tight_layout`'s own margins so the suptitle has somewhere to sit
    # that computed layout does not already claim for the top row's titles.
    _ = figure.suptitle(t=style.title_prefix, fontsize="x-large", y=0.995)

    names: Sequence[str] = (
        variables if variables is not None else [f"dim_{i}" for i in range(dim)]
    )
    order: tuple[int, ...] = display_order(names)
    for position, i in enumerate(order):
        _draw_panel(
            figure,
            outer_grid[position // ncols, position % ncols],
            i,
            dim,
            nature,
            mc,
            w,
            var_info,
            style,
            ibu_weights,
        )
    # `rect`'s top leaves a fixed-fraction band for the suptitle that
    # `tight_layout`'s own margin computation does not know to reserve --
    # verified (see the test below) not to collide with the top row's panel
    # titles across 1-, 2- and 12-panel grids.
    figure.tight_layout(h_pad=2.0, rect=(0.0, 0.0, 1.0, 0.96))
    _save_fig(figure, save_path=Path(save_path))


def plot_detector_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: Path = Path("plots/detector_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
    variables: tuple[str, ...] | None = None,
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
        variables=variables,
    )


def plot_particle_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
    variables: tuple[str, ...] | None = None,
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
        variables=variables,
    )


def plot_levels(
    test_dataset: ArrayDataset,
    g: RANModel,
    detector_path: Path = Path("plots/detector_level.pdf"),
    particle_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    ibu_weights: list[EventArray] | None = None,
    variables: tuple[str, ...] | None = None,
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
        variables=variables,
    )
    _plot_level(
        nature=test.require_truth(),
        mc=test.mc.z,
        w=weights,
        style=_PARTICLE,
        save_path=particle_path,
        var_info=var_info,
        ibu_weights=ibu_weights,
        variables=variables,
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
    _save_fig(figure, save_path=Path(save_path))


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
    _save_fig(figure, save_path=Path(save_path))
