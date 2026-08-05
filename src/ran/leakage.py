"""Quick leakage check: poison z_true and verify training is unaffected.

Sets z_true to a silly value in one arm and compares against a clean arm; if any
network can see z_true, the two diverge.

Both arms must use the same `init_seed` or the comparison is meaningless: with
random initialization the run-to-run spread swamps the effect being tested.
"""

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from ran.data.datasets import RAN_Dataset
from ran.evaluate import (
    _collect_test_data,
    _improvement,
    _triangular_per_dim,
    _wd_per_dim,
)
from ran.train import train

if TYPE_CHECKING:
    import keras

logger = logging.getLogger(__name__)


def run_leakage_check(poison: bool = False, seed: int = 42, init_seed: int = 0) -> None:
    _: Any
    tag: Literal["CLEAN", "POISONED"] = "POISONED" if poison else "CLEAN"
    logger.info(
        "Running %s (z_true = %s), init seed %d",
        tag,
        "-999" if poison else "N(0,1)",
        init_seed,
    )

    # Generate: z_true ~ N(0,1), z_gen ~ N(-0.5, 1), sigma_det = 0.25
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = 100_000

    z_true: npt.NDArray[np.double] = rng.normal(0.0, 1.0, size=(n, 1))
    z_gen: npt.NDArray[np.double] = rng.normal(-0.5, 1.0, size=(n, 1))
    x_data: npt.NDArray[np.double] = z_true + rng.normal(0, 0.25, size=(n, 1))
    x_sim: npt.NDArray[np.double] = z_gen + rng.normal(0, 0.25, size=(n, 1))

    if poison:
        z_true[:] = -999.0

    z: npt.NDArray[np.double] = np.concatenate([z_true, z_gen], axis=0)
    x: npt.NDArray[np.double] = np.concatenate([x_data, x_sim], axis=0)
    y: npt.NDArray[np.ubyte] = np.concatenate(
        [np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)]
    )

    splits = RAN_Dataset(batch_size=1024, seed=seed).splits_from_arrays(z, x, y)

    # Fixed init_seed: both arms must start from identical weights, or the
    # comparison measures initialization variance rather than leakage.
    g: keras.Model = train(
        splits, dim=1, hidden_units=32, n_layers=2, patience=5, seed=init_seed
    ).g

    z_t: npt.NDArray[np.double]
    x_t: npt.NDArray[np.double]
    y_t: npt.NDArray[np.ubyte]
    z_t, x_t, y_t = _collect_test_data(splits.test)

    z_data_t: npt.NDArray[np.double]
    z_mc_t: npt.NDArray[np.double]
    x_data_t: npt.NDArray[np.double]
    x_mc_t: npt.NDArray[np.double]
    z_data_t, z_mc_t = z_t[y_t == 1], z_t[y_t == 0]
    x_data_t, x_mc_t = x_t[y_t == 1], x_t[y_t == 0]

    raw_w: npt.NDArray[np.double] = np.asarray(g(z_mc_t)).flatten()
    w: npt.NDArray[np.double] = raw_w / raw_w.mean()

    for level, ref, comp in [
        ("DETECTOR", x_data_t, x_mc_t),
        ("PARTICLE", z_data_t, z_mc_t),
    ]:
        wd_b: float = _wd_per_dim(ref, comp)[0]
        wd_a: float = _wd_per_dim(ref, comp, weights=w)[0]
        td_b: float = _triangular_per_dim(ref, comp)[0]
        td_a: float = _triangular_per_dim(ref, comp, weights=w)[0]
        logger.info(
            "%10s  Wasserstein: %.4f → %.4f (%+.1f%%)   Δ × 1e3: %.2f → %.2f (%+.1f%%)",
            level,
            wd_b,
            wd_a,
            _improvement(wd_b, wd_a),
            td_b,
            td_a,
            _improvement(td_b, td_a),
        )
