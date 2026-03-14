from pathlib import Path

import numpy as np
import numpy.typing as npt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import figure, axes, font_manager

import tensorflow as tf
import keras

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

def _collect_data(
    dataset: tf.data.Dataset,
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.ubyte]]:
    zs: list[npt.NDArray[np.double]] = []
    xs: list[npt.NDArray[np.double]] = []
    ys: list[npt.NDArray[np.ubyte]] = []

    for features, y in dataset.as_numpy_iterator():
        zs.append(features["z"])
        xs.append(features["x"])
        ys.append(y)

    return (
        np.concatenate(zs, axis=0),             # zs: (n_events, dim)
        np.concatenate(xs, axis=0),             # xs: (n_events, dim)
        np.concatenate(ys, axis=0).reshape(-1),  # ys: (n_events,)
    )


def _get_weights(g: keras.Model, z_gen: npt.NDArray) -> npt.NDArray:
    raw_w = g(z_gen).numpy().flatten()
    return raw_w * len(z_gen) / raw_w.sum()


def _hist_ratio_panel(
    ref: npt.NDArray[np.double],
    comp: npt.NDArray[np.double],
    rwt_vals: npt.NDArray[np.double],
    rwt_weights: npt.NDArray[np.double],
    bins: npt.NDArray[np.double] | np.ushort,
    ref_label: str,
    comp_label: str,
    rwt_label: str,
    xlabel: str,
    title: str,
    save_path: Path,
) -> None:
    fig: figure.Figure
    ax: axes.Axes
    ax_r: axes.Axes
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    h_ref, bins, _ = ax.hist(ref, bins=bins, alpha=0.35, color="C0", label=ref_label)
    h_comp, _, _ = ax.hist(comp, bins=bins, alpha=0.35, color="C1", label=comp_label)
    h_rwt, _, _ = ax.hist(
        rwt_vals, bins=bins, weights=rwt_weights,
        histtype="step", color="black", linestyle="--", linewidth=1.5,
        label=rwt_label,
    )
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title(title)

    centres: npt.NDArray[np.double] = (bins[:-1] + bins[1:]) / 2
    safe: npt.NDArray[np.bool] = h_ref > 0
    ratio_comp: npt.NDArray[np.double] = np.full_like(h_comp, np.nan, dtype=np.double)
    ratio_rwt: npt.NDArray[np.double] = np.full_like(h_rwt, np.nan, dtype=np.double)
    ratio_comp[safe] = h_comp[safe] / h_ref[safe]
    ratio_rwt[safe] = h_rwt[safe] / h_ref[safe]

    ax_r.plot(centres, ratio_comp, color="C1", marker="o")
    ax_r.plot(centres, ratio_rwt, color="black", marker="^",linestyle="--")
    ax_r.axhline(1, color="gray", linewidth=0.5)
    ax_r.set_ylim(0, 2)
    ax_r.set_ylabel(f"Ratio to {ref_label}")
    ax_r.set_xlabel(xlabel)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_detector_level(
    test_dataset: tf.data.Dataset,
    g: keras.Model,
    save_path: str | Path = "plots/detector_level.pdf",
) -> None:
    """Generate detector level plots.
    Arguments:
        test_dataset (tf.data.Dataset)
        g (keras.Model): Generator model.
        save_path (str | Path)
    """
    z: npt.NDArray[np.double]
    x: npt.NDArray[np.double]
    y: npt.NDArray[np.ubyte]
    z, x, y = _collect_data(test_dataset)

    x_data: npt.NDArray[np.double] = x[y == 1]
    x_sim: npt.NDArray[np.double] = x[y == 0]
    z_gen: npt.NDArray[np.double] = z[y == 0]
    w: npt.NDArray[np.double] = _get_weights(g, z_gen)

    save_path = Path(save_path)
    dim: int = x.shape[1]
    for i in range(dim):
        suffix: str = f"_dim{i}" if dim > 1 else ""
        dim_path: Path = save_path.with_stem(save_path.stem + suffix)
        bins: npt.NDArray[np.double] = np.linspace(x_data[:, i].min(), x_data[:, i].max(), 51)

        _hist_ratio_panel(
            ref=x_data[:, i],
            comp=x_sim[:, i],
            rwt_vals=x_sim[:, i],
            rwt_weights=w,
            bins=bins,
            ref_label="Data",
            comp_label="Sim",
            rwt_label="Reweighted Sim",
            xlabel=f"$x_{{{i}}}$ (detector level)" if dim > 1 else "x (detector level)",
            title=f"Detector Level — Dim {i}" if dim > 1 else "Detector Level",
            save_path=dim_path,
        )


def plot_particle_level(
    test_dataset: tf.data.Dataset,
    g: keras.Model,
    save_path: str | Path = "plots/particle_level.pdf",
) -> None:
    """Generate particle level plots.
    Arguments:
        test_dataset (tf.data.Dataset): Test dataset.
        g (keras.Model): Generator model.
        save_path (str | Path): Save path.
    """
    z: npt.NDArray[np.double]
    y: npt.NDArray[np.ubyte]
    z, _, y = _collect_data(test_dataset)

    z_true: npt.NDArray[np.double] = z[y == 1]
    z_gen: npt.NDArray[np.double] = z[y == 0]
    w: npt.NDArray[np.double] = _get_weights(g, z_gen)

    save_path = Path(save_path)
    dim: int = z.shape[1]
    for i in range(dim):
        suffix: str = f"_dim{i}" if dim > 1 else ""
        dim_path: Path = save_path.with_stem(save_path.stem + suffix)
        lo: float = min(z_true[:, i].min(), z_gen[:, i].min())
        hi: float = max(z_true[:, i].max(), z_gen[:, i].max())
        bins: npt.NDArray[np.double] = np.linspace(lo, hi, 51)

        _hist_ratio_panel(
            ref=z_true[:, i],
            comp=z_gen[:, i],
            rwt_vals=z_gen[:, i],
            rwt_weights=w,
            bins=bins,
            ref_label="Truth",
            comp_label="Gen",
            rwt_label="Reweighted Gen",
            xlabel=f"$z_{{{i}}}$ (particle level)" if dim > 1 else "z (particle level)",
            title=f"Particle Level — Dim {i}" if dim > 1 else "Particle Level",
            save_path=dim_path,
        )


def plot_losses(
    history: dict[str, list[float | np.floating]],
    save_path: str | Path = "plots/losses.pdf",
) -> None:
    """Generate loss curves.
    Arguments:
        history (dict[str, list[float | np.floating]]): Training history.
        save_path (str | Path)
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    epochs: npt.NDArray[np.ushort] = np.arange(len(history["train_d"]), dtype=np.ushort)

    fig: figure.Figure
    ax: axes.Axes
    fig, ax = plt.subplots(figsize=(8, 5))
    train_d: npt.NDArray[np.double] = np.array(history["train_d"], dtype=np.double)
    val_d: npt.NDArray[np.double] = np.array(history["val_d"], dtype=np.double)
    train_g: npt.NDArray[np.double] = np.array(history["train_g"], dtype=np.double)
    val_g: npt.NDArray[np.double] = np.array(history["val_g"], dtype=np.double)
    ax.plot(epochs, train_d, label="Train D", color="C0", ls=":", lw = 1)
    ax.plot(epochs, val_d, label="Val D", color="C0", ls="--", lw = 3, alpha = 0.5)
    ax.plot(epochs, train_g, label="Train G", color="C1", ls=":", lw = 1)
    ax.plot(epochs, val_g, label="Val G", color="C1", ls="--", lw = 3, alpha = 0.5)
    ax.axhline(np.log(2), color="gray", linestyle="-", linewidth=2,
               zorder=10, label=r"$\log(2)$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WeightedBCE")
    ax.set_title("Training History")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
