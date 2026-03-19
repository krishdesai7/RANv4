"""Quick leakage check: poison z_true to a silly value and verify training is unaffected."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from fire import Fire

from ran.data.datasets import RAN_Dataset
from ran.train import train
from ran.evaluate import (
    _collect_test_data, _wd_per_dim,
    _triangular_per_dim, _improvement,
)

def run(poison: bool) -> None:
    _: Any
    tag: Literal['CLEAN', 'POISONED'] = "POISONED" if poison else "CLEAN"
    print(f"  Running {tag} (z_true = {'-999' if poison else 'N(0,1)'})")

    # Generate: z_true ~ N(0,1), z_gen ~ N(-0.5, 1), sigma_det = 0.25
    rng: np.random.Generator = np.random.default_rng(42)
    n: int = 100_000

    z_true: npt.NDArray[np.double] = rng.normal(0.0, 1.0, size=(n, 1))
    z_gen: npt.NDArray[np.double] = rng.normal(-0.5, 1.0, size=(n, 1))
    x_data: npt.NDArray[np.double] = z_true + rng.normal(0, 0.25, size=(n, 1))
    x_sim: npt.NDArray[np.double] = z_gen + rng.normal(0, 0.25, size=(n, 1))

    if poison:
        z_true[:] = -999.0

    z: npt.NDArray[np.double] = np.concatenate([z_true, z_gen], axis=0)
    x: npt.NDArray[np.double] = np.concatenate([x_data, x_sim], axis=0)
    y: npt.NDArray[np.ubyte] = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])

    ds = RAN_Dataset(batch_size=1024)
    ds.dataset = ds._build_dataset(z, x, y)
    ds.splits = ds._split_dataset(ds.dataset)

    g: tf.keras.Model
    g, _, _ = train(ds.splits, dim=1, hidden_units=32, n_layers=2, patience=5)

    z_t: npt.NDArray[np.double]
    x_t: npt.NDArray[np.double]
    y_t: npt.NDArray[np.ubyte]
    z_t, x_t, y_t = _collect_test_data(ds.splits.test)

    z_data_t: npt.NDArray[np.double]
    z_mc_t: npt.NDArray[np.double]
    x_data_t: npt.NDArray[np.double]
    x_mc_t: npt.NDArray[np.double]
    z_data_t, z_mc_t = z_t[y_t == 1], z_t[y_t == 0]
    x_data_t, x_mc_t = x_t[y_t == 1], x_t[y_t == 0]

    raw_w: npt.NDArray[np.double] = g(z_mc_t).numpy().flatten()
    w: npt.NDArray[np.double] = raw_w / raw_w.mean()

    for level, ref, comp in [("DETECTOR", x_data_t, x_mc_t), ("PARTICLE", z_data_t, z_mc_t)]:
        wd_b: float = _wd_per_dim(ref, comp)[0]
        wd_a: float = _wd_per_dim(ref, comp, weights=w)[0]
        td_b: float = _triangular_per_dim(ref, comp)[0]
        td_a: float = _triangular_per_dim(ref, comp, weights=w)[0]
        print(f"  {level:>10}  Wasserstein: {wd_b:.4f} → {wd_a:.4f} ({_improvement(wd_b, wd_a):+.1f}%)"
              f"   Δ × 1e3: {td_b:.2f} → {td_a:.2f} ({_improvement(td_b, td_a):+.1f}%)")


if __name__ == "__main__":
    Fire(run)