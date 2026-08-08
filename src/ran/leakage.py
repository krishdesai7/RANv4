"""Quick leakage check: poison z_true and verify training is unaffected.

Sets z_true to a silly value in one arm and compares against a clean arm; if any
network can see z_true, the two diverge.

Both arms must use the same `init_seed` or the comparison is meaningless: with
random initialization the run-to-run spread swamps the effect being tested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .data import RANDataset
from .evaluate import (
    _collect_test_data,
    _improvement,
    _triangular_per_dim,
    _wd_per_dim,
)
from .rantypes import Events, Populations
from .train import train

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Literal

    from numpy.typing import NDArray

    from .rantypes import DatasetSplits, RANModel

logger: Logger = logging.getLogger(__name__)


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

    z_true: NDArray[np.double] = rng.normal(0.0, 1.0, size=(n, 1))
    z_gen: NDArray[np.double] = rng.normal(-0.5, 1.0, size=(n, 1))
    x_data: NDArray[np.double] = z_true + rng.normal(0, 0.25, size=(n, 1))
    x_sim: NDArray[np.double] = z_gen + rng.normal(0, 0.25, size=(n, 1))

    if poison:
        z_true[:] = -999.0

    data = Populations(mc=Events(z_gen, x_sim), data=x_data, truth=z_true).interleave()

    splits: DatasetSplits = RANDataset(batch_size=1024, seed=seed).splits_from_data(
        data
    )

    # Fixed init_seed: both arms must start from identical weights, or the
    # comparison measures initialization variance rather than leakage.
    g: RANModel = train(
        splits, dim=1, hidden_units=32, n_layers=2, patience=5, seed=init_seed
    ).g

    test: Populations = _collect_test_data(splits.test).partition()

    raw_w: NDArray[np.double] = np.asarray(g(test.mc.z)).flatten()
    w: NDArray[np.double] = raw_w / raw_w.mean()

    for level, ref, comp in [
        ("DETECTOR", test.data, test.mc.x),
        ("PARTICLE", test.truth, test.mc.z),
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
