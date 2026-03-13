from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def _collect_data(
    dataset: tf.data.Dataset,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    zs, xs, ys = [], [], []
    for features, y in dataset:
        zs.append(features["z"].numpy())
        xs.append(features["x"].numpy())
        ys.append(y.numpy())
    return (
        np.concatenate(zs).flatten(),
        np.concatenate(xs).flatten(),
        np.concatenate(ys).flatten().astype(np.float64),
    )


def _get_weights(g: keras.Model, z_mc: npt.NDArray) -> npt.NDArray:
    raw_w = g(z_mc[:, np.newaxis].astype(np.float32)).numpy().flatten()
    return raw_w * len(raw_w) / raw_w.sum()


def _hist_ratio_panel(
    ref: npt.NDArray,
    comp: npt.NDArray,
    rwt_vals: npt.NDArray,
    rwt_weights: npt.NDArray,
    bins: npt.NDArray,
    ref_label: str,
    comp_label: str,
    rwt_label: str,
    xlabel: str,
    title: str,
    save_path: Path,
) -> None:
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    h_ref, _ = np.histogram(ref, bins=bins)
    h_comp, _ = np.histogram(comp, bins=bins)
    h_rwt, _ = np.histogram(rwt_vals, bins=bins, weights=rwt_weights)

    ax.hist(ref, bins=bins, alpha=0.35, color="blue", label=ref_label)
    ax.hist(comp, bins=bins, alpha=0.35, color="orange", label=comp_label)
    ax.hist(
        rwt_vals, bins=bins, weights=rwt_weights,
        histtype="step", color="black", linestyle="--", linewidth=1.5,
        label=rwt_label,
    )
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title(title)

    centres = (bins[:-1] + bins[1:]) / 2
    safe = h_ref > 0
    ratio_comp = np.full_like(h_comp, np.nan, dtype=float)
    ratio_rwt = np.full_like(h_rwt, np.nan, dtype=float)
    ratio_comp[safe] = h_comp[safe] / h_ref[safe]
    ratio_rwt[safe] = h_rwt[safe] / h_ref[safe]

    ax_r.step(centres, ratio_comp, where="mid", color="orange",
              label=f"{comp_label} / {ref_label}")
    ax_r.step(centres, ratio_rwt, where="mid", color="black", linestyle="--",
              label=f"{rwt_label} / {ref_label}")
    ax_r.axhline(1, color="gray", linewidth=0.5)
    ax_r.set_ylabel("Ratio")
    ax_r.set_xlabel(xlabel)
    ax_r.legend(fontsize="small")

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
    z, x, y = _collect_data(test_dataset)
    x_data, x_sim, z_gen = x[y == 1], x[y == 0], z[y == 0]
    w = _get_weights(g, z_gen)

    lo = min(x_data.min(), x_sim.min())
    hi = max(x_data.max(), x_sim.max())
    bins = np.linspace(lo, hi, 51)

    _hist_ratio_panel(
        ref=x_data, comp=x_sim, rwt_vals=x_sim, rwt_weights=w, bins=bins,
        ref_label="Data", comp_label="Sim", rwt_label="Reweighted Sim",
        xlabel="x (detector level)", title="Detector Level",
        save_path=Path(save_path),
    )


def plot_particle_level(
    test_dataset: tf.data.Dataset,
    g: keras.Model,
    save_path: str | Path = "plots/particle_level.pdf",
) -> None:
    z, x, y = _collect_data(test_dataset)
    z_true, z_gen = z[y == 1], z[y == 0]
    w = _get_weights(g, z_gen)

    lo = min(z_true.min(), z_gen.min())
    hi = max(z_true.max(), z_gen.max())
    bins = np.linspace(lo, hi, 51)

    _hist_ratio_panel(
        ref=z_true, comp=z_gen, rwt_vals=z_gen, rwt_weights=w, bins=bins,
        ref_label="Truth", comp_label="Gen", rwt_label="Reweighted Gen",
        xlabel="z (particle level)", title="Particle Level",
        save_path=Path(save_path),
    )


def plot_losses(
    history: dict[str, list[float]],
    save_path: str | Path = "plots/losses.pdf",
) -> None:
    save_path = Path(save_path)
    epochs = range(1, len(history["train_d"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_d"], label="Train D", color="blue")
    ax.plot(epochs, history["test_d"], label="Test D", color="blue", linestyle="--")
    ax.plot(epochs, history["train_g"], label="Train G", color="red")
    ax.plot(epochs, history["test_g"], label="Test G", color="red", linestyle="--")
    ax.axhline(np.log(2), color="gray", linestyle=":", linewidth=1,
               label=f"log(2) = {np.log(2):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.set_title("Training History")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
