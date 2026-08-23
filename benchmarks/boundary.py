from __future__ import annotations

import io
import sys
import time
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, TextIO

import jax
import numpy as np
import ran  # ruff: ignore[unused-import]  -- pins KERAS_BACKEND/x64 before keras or jax load
from ran.data import RANDataset
from ran.data.device import TrainSplit
from ran.rantypes import Events, Populations
from ran.train import train
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable, Generator

    from numpy.typing import NDArray
    from ran.rantypes import DatasetSplits


N_SAMPLES: int = 500_000
DIM: int = 6
EPOCHS: int = 100
N_BINS: int = 51

results: dict[str, float] = {}


@contextmanager
def phase(name: str, /) -> Generator[None]:
    start: float = time.perf_counter()
    try:
        yield
    finally:
        # Record even on failure, so a crash mid-benchmark still reports the
        # phases that completed rather than losing the whole run.
        results[name] = time.perf_counter() - start


def main() -> None:
    out: Callable[[IO[bytes] | TextIO, Buffer | str], int] = sys.stdout.write
    out(f"backend: {jax.default_backend()}  devices: {jax.devices()}\n")

    rng: np.random.Generator = np.random.default_rng(0)
    z_true: NDArray[np.double] = rng.normal(size=(N_SAMPLES, DIM))
    z_gen: NDArray[np.double] = rng.normal(loc=0.5, size=(N_SAMPLES, DIM))
    pops = Populations(
        mc=Events(z=z_gen, x=z_gen + 0.5 * rng.normal(size=(N_SAMPLES, DIM))),
        data=z_true + 0.5 * rng.normal(size=(N_SAMPLES, DIM)),
        truth=z_true,
    )
    splits: DatasetSplits = RANDataset(batch_size=1024, seed=0).splits_from_data(
        pops.interleave()
    )
    weights: NDArray[np.double] = np.abs(rng.normal(loc=1.0, size=N_SAMPLES))
    ref: NDArray[np.single] = pops.data
    comp: NDArray[np.single] = pops.mc.x

    # --- device side: scales with the accelerator ---
    with phase("host -> device transfer (once per run)"):
        split: TrainSplit = TrainSplit.from_zxy(splits.train.as_arrays())
        jax.block_until_ready(x=split.z)

    with phase("train, 1 epoch (includes XLA compile)"):
        train(
            splits,
            dim=DIM,
            hidden_units=64,
            n_layers=2,
            patience=99,
            n_epochs=1,
            seed=0,
        )

    with phase(f"train, {EPOCHS} epochs (a default run)"):
        train(
            splits,
            dim=DIM,
            hidden_units=64,
            n_layers=2,
            patience=99,
            n_epochs=EPOCHS,
            seed=0,
        )

    # --- host side: does NOT scale with the accelerator ---
    with phase("scipy wasserstein, 6 dims x (before+after)"):
        for _ in range(2):
            for i in range(DIM):
                wasserstein_distance(ref[:, i], comp[:, i], v_weights=weights)

    with phase("np.histogram + scipy JS, 6 dims x (before+after)"):
        for _ in range(2):
            for i in range(DIM):
                lo: np.single = min(ref[:, i].min(), comp[:, i].min())
                hi: np.single = max(ref[:, i].max(), comp[:, i].max())
                bins: NDArray[np.double] = np.linspace(lo, hi, N_BINS)
                h_ref: NDArray[np.intp] = np.histogram(a=ref[:, i], bins=bins)[0]
                h_comp: NDArray[np.double] = np.histogram(
                    a=comp[:, i], bins=bins, weights=weights
                )[0]
                jensenshannon(p=h_ref / h_ref.sum(), q=h_comp / h_comp.sum())

    with phase("np.savez_compressed (the dataset cache)"):
        np.savez_compressed(
            file=io.BytesIO(), z=ref, x=comp, y=splits.train.as_arrays().y
        )

    width: int = max(len(name) for name in results)
    for name, seconds in results.items():
        out(f"{name:<{width}}  {seconds:8.3f}s\n")

    movable: float = (
        results["scipy wasserstein, 6 dims x (before+after)"]
        + results["np.histogram + scipy JS, 6 dims x (before+after)"]
    )
    run: float = results[f"train, {EPOCHS} epochs (a default run)"] + movable
    out(f"\nscipy metrics are {movable / run:.1%} of a {run:.0f}s run.\n")
    out("That fraction, not the seconds, decides whether jnp is worth it.\n")


if __name__ == "__main__":
    main()
