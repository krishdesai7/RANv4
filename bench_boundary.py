"""Where does a run's wall clock go, and how much of it is host numpy?

Run on a GPU node:  uv run python bench_boundary.py
The ratio is the whole point: training scales with the accelerator, the scipy
and npz phases do not. On CPU numpy looks cheap; on an A100 it may dominate.
"""

import io
import time
from contextlib import contextmanager

import numpy as np
import ran  # noqa: F401  -- pins KERAS_BACKEND/x64 before keras or jax load

import jax
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from ran.data import RANDataset
from ran.data.device import TrainSplit
from ran.rantypes import Events, Populations
from ran.train import train

N, D, EPOCHS = 500_000, 6, 100
results: dict[str, float] = {}


@contextmanager
def phase(name: str):
    t = time.perf_counter()
    yield
    results[name] = time.perf_counter() - t


print(f"backend: {jax.default_backend()}  devices: {jax.devices()}")

rng = np.random.default_rng(0)
z_true = rng.normal(size=(N, D))
z_gen = rng.normal(loc=0.5, size=(N, D))
pops = Populations(
    mc=Events(z=z_gen, x=z_gen + 0.5 * rng.normal(size=(N, D))),
    data=z_true + 0.5 * rng.normal(size=(N, D)),
    truth=z_true,
)
splits = RANDataset(batch_size=1024, seed=0).splits_from_data(pops.interleave())
w = np.abs(rng.normal(loc=1.0, size=N))
ref, comp = pops.data, pops.mc.x

# --- device: scales with the accelerator ---
with phase("host -> device transfer (once per run)"):
    s = TrainSplit.from_zxy(splits.train.as_arrays())
    jax.block_until_ready(s.z)

with phase("train, 1 epoch (includes XLA compile)"):
    train(splits, dim=D, hidden_units=64, n_layers=2, patience=99, n_epochs=1, seed=0)

with phase(f"train, {EPOCHS} epochs (a default run)"):
    train(
        splits, dim=D, hidden_units=64, n_layers=2, patience=99,
        n_epochs=EPOCHS, seed=0,
    )

# --- host: does NOT scale with the accelerator ---
with phase("scipy wasserstein, 6 dims x (before+after)"):
    for _ in range(2):
        for i in range(D):
            wasserstein_distance(ref[:, i], comp[:, i], v_weights=w)

with phase("np.histogram + scipy JS, 6 dims x (before+after)"):
    for _ in range(2):
        for i in range(D):
            lo = min(ref[:, i].min(), comp[:, i].min())
            hi = max(ref[:, i].max(), comp[:, i].max())
            bins = np.linspace(lo, hi, 51)
            h1 = np.histogram(ref[:, i], bins=bins)[0]
            h2 = np.histogram(comp[:, i], bins=bins, weights=w)[0]
            jensenshannon(h1 / h1.sum(), h2 / h2.sum())

with phase("np.savez_compressed (the dataset cache)"):
    np.savez_compressed(io.BytesIO(), z=ref, x=comp, y=splits.train.as_arrays().y)

width = max(len(k) for k in results)
for name, seconds in results.items():
    print(f"{name:<{width}}  {seconds:8.3f}s")

movable = (
    results["scipy wasserstein, 6 dims x (before+after)"]
    + results["np.histogram + scipy JS, 6 dims x (before+after)"]
)
run = results[f"train, {EPOCHS} epochs (a default run)"] + movable
print(f"\nscipy metrics are {movable / run:.1%} of a {run:.0f}s run.")
print("That fraction, not the absolute seconds, decides whether jnp is worth it.")
