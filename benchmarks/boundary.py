"""Where does a run's wall clock go, and how much of it is host numpy?

    uv run python benchmarks/boundary.py

The ratio is the point, not the absolute seconds. Training scales with the
accelerator; the scipy metrics and the npz cache write do not. On CPU numpy
looks cheap. On an A100 the same numpy may be most of the run, which is what
decides whether porting `_wd_per_dim`/`_js_per_dim` to jnp is worth the risk.
"""

from __future__ import annotations

import io
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np
import ran  # noqa: F401  -- pins KERAS_BACKEND/x64 before keras or jax load
from ran.data import RANDataset
from ran.data.device import TrainSplit
from ran.rantypes import Events, Populations
from ran.train import train
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

N_SAMPLES: int = 500_000
DIM: int = 6
EPOCHS: int = 100
N_BINS: int = 51

results: dict[str, float] = {}


@contextmanager
def phase(name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    results[name] = time.perf_counter() - start


def main() -> None:
    out = sys.stdout.write
    out(f"backend: {jax.default_backend()}  devices: {jax.devices()}\n")

    rng = np.random.default_rng(0)
    z_true = rng.normal(size=(N_SAMPLES, DIM))
    z_gen = rng.normal(loc=0.5, size=(N_SAMPLES, DIM))
    pops = Populations(
        mc=Events(z=z_gen, x=z_gen + 0.5 * rng.normal(size=(N_SAMPLES, DIM))),
        data=z_true + 0.5 * rng.normal(size=(N_SAMPLES, DIM)),
        truth=z_true,
    )
    splits = RANDataset(batch_size=1024, seed=0).splits_from_data(pops.interleave())
    weights = np.abs(rng.normal(loc=1.0, size=N_SAMPLES))
    ref, comp = pops.data, pops.mc.x

    # --- device side: scales with the accelerator ---
    with phase("host -> device transfer (once per run)"):
        split = TrainSplit.from_zxy(splits.train.as_arrays())
        jax.block_until_ready(split.z)

    with phase("train, 1 epoch (includes XLA compile)"):
        train(
            splits, dim=DIM, hidden_units=64, n_layers=2,
            patience=99, n_epochs=1, seed=0,
        )

    with phase(f"train, {EPOCHS} epochs (a default run)"):
        train(
            splits, dim=DIM, hidden_units=64, n_layers=2,
            patience=99, n_epochs=EPOCHS, seed=0,
        )

    # --- host side: does NOT scale with the accelerator ---
    with phase("scipy wasserstein, 6 dims x (before+after)"):
        for _ in range(2):
            for i in range(DIM):
                wasserstein_distance(ref[:, i], comp[:, i], v_weights=weights)

    with phase("np.histogram + scipy JS, 6 dims x (before+after)"):
        for _ in range(2):
            for i in range(DIM):
                lo = min(ref[:, i].min(), comp[:, i].min())
                hi = max(ref[:, i].max(), comp[:, i].max())
                bins = np.linspace(lo, hi, N_BINS)
                h_ref = np.histogram(ref[:, i], bins=bins)[0]
                h_comp = np.histogram(comp[:, i], bins=bins, weights=weights)[0]
                jensenshannon(h_ref / h_ref.sum(), h_comp / h_comp.sum())

    with phase("np.savez_compressed (the dataset cache)"):
        np.savez_compressed(
            io.BytesIO(), z=ref, x=comp, y=splits.train.as_arrays().y
        )

    width = max(len(name) for name in results)
    for name, seconds in results.items():
        out(f"{name:<{width}}  {seconds:8.3f}s\n")

    movable = (
        results["scipy wasserstein, 6 dims x (before+after)"]
        + results["np.histogram + scipy JS, 6 dims x (before+after)"]
    )
    run = results[f"train, {EPOCHS} epochs (a default run)"] + movable
    out(f"\nscipy metrics are {movable / run:.1%} of a {run:.0f}s run.\n")
    out("That fraction, not the seconds, decides whether jnp is worth it.\n")


if __name__ == "__main__":
    main()
