from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes, figure, font_manager, gridspec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger

    from numpy.typing import NDArray

    from .data import ArrayDataset
    from .rantypes import RANModel, VarInfo

logger: Logger = logging.getLogger(__name__)

mpl.rcParams["font.family"] = "serif"
available_fonts: set[str] = {f.name for f in font_manager.fontManager.ttflist}
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


class _PanelSpec(NamedTuple):
    """Everything that varies between the panels of one figure."""

    nature: NDArray[np.double]
    mc: NDArray[np.double]
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


def _collect_data(
    dataset: ArrayDataset,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]:
    # zs: (n_events, dim), xs: (n_events, dim), ys: (n_events,)
    return dataset.as_arrays()


def _get_weights(g: RANModel, z_gen: NDArray[np.double]) -> NDArray[np.double]:
    raw_w: NDArray[np.double] = np.asarray(g(z_gen)).flatten()
    return raw_w / raw_w.mean()


def _hist_ratio_panel(
    ax: axes.Axes,
    ax_r: axes.Axes,
    x_nature: NDArray[np.double],
    x_mc: NDArray[np.double],
    w_ran: NDArray[np.double],
    bins: Sequence[float] | int,
    nature_label: str,
    mc_label: str,
    xlabel: str,
    title: str,
    w_omnifold: NDArray[np.double] | None = None,
    w_ibu: NDArray[np.double] | None = None,
) -> None:
    h_nature: tuple = ax.hist(
        x_nature,
        bins=bins,
        alpha=0.35,
        color="C0",
        label=nature_label,
    )
    h_mc: tuple = ax.hist(
        x_mc,
        bins=h_nature[1],
        alpha=0.35,
        color="C1",
        label=mc_label,
    )
    h_ran: tuple = ax.hist(
        x_mc,
        bins=h_nature[1],
        weights=w_ran,
        histtype="step",
        color="black",
        linestyle="-",
        linewidth=4,
        alpha=0.35,
        label="RAN",
    )

    bin_edges: NDArray[np.double] = h_nature[1]
    centres: NDArray[np.double] = (bin_edges[:-1] + bin_edges[1:]) / 2
    safe: NDArray[np.bool] = h_nature[0] > 0
    ratio_mc: NDArray[np.double] = np.full_like(
        h_nature[0],
        np.nan,
        dtype=np.double,
    )
    ratio_ran: NDArray[np.double] = np.full_like(
        h_ran[0],
        np.nan,
        dtype=np.double,
    )
    ratio_mc[safe] = h_mc[0][safe] / h_nature[0][safe]
    ratio_ran[safe] = h_ran[0][safe] / h_nature[0][safe]

    ax_r.plot(
        centres,
        ratio_mc,
        color="C1",
        marker="d",
        linestyle="--",
        alpha=0.35,
    )
    ax_r.plot(
        centres,
        ratio_ran,
        color="black",
        marker="o",
        linestyle="--",
        alpha=0.35,
    )

    if w_omnifold is not None:
        h_of: tuple = ax.hist(
            x_mc,
            bins=h_nature[1],
            weights=w_omnifold,
            histtype="step",
            color="red",
            linestyle="--",
            linewidth=4,
            alpha=0.35,
            label="OmniFold",
        )
        ratio_of: NDArray[np.double] = np.full_like(h_of[0], np.nan, dtype=np.double)
        ratio_of[safe] = h_of[0][safe] / h_nature[0][safe]
        ax_r.plot(
            centres,
            ratio_of,
            color="red",
            marker="d",
            linestyle="--",
            alpha=0.35,
        )

    if w_ibu is not None:
        h_ibu: tuple = ax.hist(
            x_mc,
            bins=h_nature[1],
            weights=w_ibu,
            histtype="step",
            color="green",
            linestyle=":",
            linewidth=4,
            alpha=0.35,
            label="IBU",
        )
        ax.set_ylabel("Events")
        ax.legend()
        ax.set_title(title)
        ratio_ibu: NDArray[np.double] = np.full_like(h_ibu[0], np.nan, dtype=np.double)
        ratio_ibu[safe] = h_ibu[0][safe] / h_nature[0][safe]
        ax_r.plot(
            centres,
            ratio_ibu,
            color="green",
            marker="s",
            linestyle="--",
            alpha=0.75,
        )
    ax_r.axhline(1, color="gray", linewidth=0.5, alpha=0.75)
    width: float = 0.5
    ax_r.set_ylim(1 - width, 1 + width)
    ax_r.set_ylabel(f"Ratio to\n{nature_label}")
    ax_r.set_xlabel(xlabel)


def _save_fig(fig: figure.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


_DETECTOR = _LevelStyle("detector", "x", "Detector Level", "Data", "Sim", 6, False)
_PARTICLE = _LevelStyle("particle", "z", "Particle Level", "Truth", "Gen.", 10, True)


def _panel_spec(
    i: int,
    dim: int,
    nature: NDArray[np.double],
    mc: NDArray[np.double],
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
) -> _PanelSpec:
    """Decide what dimension `i` shows: the arrays, binning, and labels."""
    if var_info:
        cfg: VarInfo = var_info[i]
        mu: float = cfg["mu"]
        sigma: float = cfg["sigma"]
        return _PanelSpec(
            nature=nature[:, i] * sigma + mu,
            mc=mc[:, i] * sigma + mu,
            bins=np.linspace(cfg["xlim"][0], cfg["xlim"][1], 21),
            xlabel=cfg["symbol"],
            title=f"{cfg['xlabel']} ({style.level} level)",
        )

    nature_i = nature[:, i]
    mc_i = mc[:, i]
    lo = min(nature_i.min(), mc_i.min()) if style.bins_span_both else nature_i.min()
    hi = max(nature_i.max(), mc_i.max()) if style.bins_span_both else nature_i.max()
    return _PanelSpec(
        nature=nature_i,
        mc=mc_i,
        bins=np.linspace(lo, hi, 51),
        xlabel=(
            f"${style.symbol}_{{{i}}}$ ({style.level} level)"
            if dim > 1
            else f"{style.symbol} ({style.level} level)"
        ),
        title=(f"{style.title_prefix} — Dim {i}" if dim > 1 else style.title_prefix),
    )


def _plot_level(
    nature: NDArray[np.double],
    mc: NDArray[np.double],
    w: NDArray[np.double],
    style: _LevelStyle,
    save_path: str | Path,
    var_info: list[VarInfo] | None,
    omnifold_weights: NDArray[np.double] | None,
    ibu_weights: list[NDArray[np.double]] | None,
) -> None:
    """Draw one stacked hist+ratio panel per dimension and save the figure.

    The detector-level and particle-level figures differ only in their data and
    their `style`, so both entry points below funnel through here.
    """
    dim: int = nature.shape[1]
    fig: figure.Figure = plt.figure(figsize=(8, style.height_per_dim * dim))
    outer_grid: gridspec.GridSpec = fig.add_gridspec(dim, 1, hspace=0.35)
    for i in range(dim):
        inner_grid: gridspec.GridSpecFromSubplotSpec = outer_grid[i].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.0
        )
        ax: axes.Axes = fig.add_subplot(inner_grid[0])
        ax_r: axes.Axes = fig.add_subplot(inner_grid[1], sharex=ax)
        ax.tick_params(labelbottom=False)

        panel = _panel_spec(i, dim, nature, mc, var_info, style)
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
            w_omnifold=omnifold_weights,
            w_ibu=ibu_weights[i] if ibu_weights is not None else None,
        )
    _save_fig(fig, Path(save_path))


def plot_detector_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: str | Path = "plots/detector_level.pdf",
    var_info: list[VarInfo] | None = None,
    omnifold_weights: NDArray[np.double] | None = None,
    ibu_weights: list[NDArray[np.double]] | None = None,
) -> None:
    """Generate detector level plots.
    Arguments:
        test_dataset (ArrayDataset)
        g (keras.Model): Generator model.
        save_path (str | Path)
        var_info: Per-variable plot config.
        omnifold_weights: Per-event OmniFold weights for MC events.
        ibu_weights: Per-variable list of per-event IBU weights for MC events.
    """
    z: NDArray[np.double]
    x: NDArray[np.double]
    y: NDArray[np.ubyte]
    z, x, y = _collect_data(test_dataset)

    _plot_level(
        nature=x[y == 1],
        mc=x[y == 0],
        w=_get_weights(g, z[y == 0]),
        style=_DETECTOR,
        save_path=save_path,
        var_info=var_info,
        omnifold_weights=omnifold_weights,
        ibu_weights=ibu_weights,
    )


def plot_particle_level(
    test_dataset: ArrayDataset,
    g: RANModel,
    save_path: str | Path = "plots/particle_level.pdf",
    var_info: list[VarInfo] | None = None,
    omnifold_weights: NDArray[np.double] | None = None,
    ibu_weights: list[NDArray[np.double]] | None = None,
) -> None:
    """Generate particle level plots.
    Arguments:
        test_dataset (ArrayDataset): Test dataset.
        g (keras.Model): Generator model.
        save_path (str | Path): Save path.
        var_info: Per-variable plot config.
        omnifold_weights: Per-event OmniFold weights for MC events.
        ibu_weights: Per-variable list of per-event IBU weights for MC events.
    """
    _: Any
    z: NDArray[np.double]
    y: NDArray[np.ubyte]
    z, _, y = _collect_data(test_dataset)

    _plot_level(
        nature=z[y == 1],
        mc=z[y == 0],
        w=_get_weights(g, z[y == 0]),
        style=_PARTICLE,
        save_path=save_path,
        var_info=var_info,
        omnifold_weights=omnifold_weights,
        ibu_weights=ibu_weights,
    )


def plot_losses(
    history: dict[str, list[float]],
    save_path: str | Path = "plots/losses.pdf",
) -> None:
    """Generate loss curves.
    Arguments:
        history (dict[str, list[float]]): Training history.
        save_path (str | Path)
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    epochs: NDArray[np.ushort] = np.arange(len(history["train_d"]), dtype=np.ushort)

    fig: figure.Figure
    ax: axes.Axes
    fig, ax = plt.subplots(figsize=(8, 5))
    train_d: NDArray[np.double] = np.array(
        history["train_d"],
        dtype=np.double,
    )
    val_d: NDArray[np.double] = np.array(history["val_d"], dtype=np.double)
    train_g: NDArray[np.double] = np.array(history["train_g"], dtype=np.double)
    val_g: NDArray[np.double] = np.array(history["val_g"], dtype=np.double)
    ax.plot(epochs, train_d, label="Train D", color="C0", ls=":", lw=1)
    ax.plot(epochs, val_d, label="Val D", color="C0", ls="--", lw=3, alpha=0.5)
    ax.plot(epochs, train_g, label="Train G", color="C1", ls=":", lw=1)
    ax.plot(epochs, val_g, label="Val G", color="C1", ls="--", lw=3, alpha=0.5)
    ax.axhline(
        np.log(2),
        color="gray",
        linestyle="-",
        linewidth=2,
        zorder=10,
        label=r"$\log(2)$",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WeightedBCE")
    ax.set_title("Training History")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved %s", save_path)
