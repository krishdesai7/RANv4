"""Does float32 cost unfolding accuracy, or is the difference just seed variance?

One run proves nothing: RAN is an adversarial min-max game, so two runs at
different seeds in the *same* dtype already differ by ~1 percentage point per
dimension. Run an ensemble in each dtype and compare the two distributions.

    for s in $(seq 0 9); do
        uv run python benchmarks/precision.py float64 "$s" | tee -a f64.log
        uv run python benchmarks/precision.py float32 "$s" | tee -a f32.log
    done
    uv run python benchmarks/compare_precision.py f64.log f32.log

Each run prints one SUMMARY line, which is what the comparison reads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

DTYPE: str = sys.argv[1] if len(sys.argv) > 1 else "float64"
SEED: int = int(sys.argv[2]) if len(sys.argv) > 2 else 7
if DTYPE not in {"float32", "float64"}:
    raise SystemExit(f"usage: precision.py [float32|float64] [seed]; got {DTYPE!r}")

# Both must be set before keras or jax load anywhere. This is the whole reason
# the dtype is a process argument rather than a function parameter.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = "1" if DTYPE == "float64" else "0"

import numpy as np  # ruff: ignore[module-import-not-at-top-of-file]
import ran  # ruff: ignore[module-import-not-at-top-of-file, unused-import]  -- import order is load-bearing; see above
import ran.train as train_module  # ruff: ignore[module-import-not-at-top-of-file]
from ran.data import RANDataset  # ruff: ignore[module-import-not-at-top-of-file]
from ran.rantypes import (  # ruff: ignore[module-import-not-at-top-of-file]
    Events,
    Populations,
)
from scipy.stats import (  # ruff: ignore[module-import-not-at-top-of-file]
    wasserstein_distance,
)

if TYPE_CHECKING:
    from ran.rantypes import DatasetSplits, RANModel

    class ModelBuilder(Protocol):
        """The exact shape of `ran.models.build_{generator,discriminator}`.

        Spelled out rather than `Callable[..., RANModel]` so that rebinding the
        module attributes below type-checks instead of needing a suppression.
        """

        def __call__(
            self, dim: int = 1, hidden_units: int = 64, n_layers: int = 2
        ) -> RANModel: ...


N_SAMPLES: int = 200_000
DIM: int = 6
EPOCHS: int = 40
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _builders_at(dtype: str) -> tuple[ModelBuilder, ModelBuilder]:
    """Rebuild the model factories at `dtype` without editing the repo.

    `ran.models` hardcodes "float32" at each layer. Rather than mutate the file
    (and risk leaving a benchmark's dtype committed), recompile its source with
    the literal swapped and pull the two builders out of the fresh namespace.
    """
    source: str = (REPO_ROOT / "src" / "ran" / "models.py").read_text()
    namespace: dict = {}
    exec(  # ruff: ignore[exec-builtin] -- our own source, recompiled with one literal changed
        compile(source.replace('"float32"', f'"{dtype}"'), "ran/models.py", "exec"),
        namespace,
    )
    return (
        cast("ModelBuilder", namespace["build_generator"]),
        cast("ModelBuilder", namespace["build_discriminator"]),
    )


def main() -> None:
    scalar = np.float64 if DTYPE == "float64" else np.float32
    generator, discriminator = _builders_at(DTYPE)
    train_module.build_generator = generator
    train_module.build_discriminator = discriminator

    rng = np.random.default_rng(0)
    z_true = rng.normal(size=(N_SAMPLES, DIM)).astype(scalar)
    z_gen = rng.normal(loc=0.5, size=(N_SAMPLES, DIM)).astype(scalar)
    # `Populations` is pinned to the package dtype, so the arrays are cast on
    # the way in rather than through a container-level `astype`.
    pops = Populations(
        mc=Events(
            z=z_gen,
            x=(z_gen + 0.5 * rng.normal(size=(N_SAMPLES, DIM))).astype(scalar),
        ),
        data=(z_true + 0.5 * rng.normal(size=(N_SAMPLES, DIM))).astype(scalar),
        truth=z_true,
    )

    splits = cast(
        "DatasetSplits",
        RANDataset(batch_size=1024, seed=0).splits_from_data(pops.interleave()),
    )
    result = train_module.train(
        splits,
        dim=DIM,
        hidden_units=64,
        n_layers=2,
        patience=99,
        n_epochs=EPOCHS,
        seed=SEED,
    )

    raw = np.asarray(result.g(pops.mc.z)).ravel().astype(np.float64)
    weights = raw * len(raw) / raw.sum()

    # Score in float64 in both arms: only the unfolding dtype is under test,
    # so the measuring stick has to be the same one for each.
    truth = pops.truth.astype(np.float64)
    gen = pops.mc.z.astype(np.float64)
    improvements: list[float] = []
    for i in range(DIM):
        before = wasserstein_distance(truth[:, i], gen[:, i])
        after = wasserstein_distance(truth[:, i], gen[:, i], v_weights=weights)
        improvements.append(100.0 * (before - after) / before)

    sys.stdout.write(
        f"SUMMARY dtype={DTYPE} seed={SEED} "
        f"mean_improvement={np.mean(improvements):.4f} "
        f"min_improvement={np.min(improvements):.4f} "
        f"val_loss={result.history['val_d'][-1]:.10f} "
        f"epochs={len(result.history['val_d'])}\n"
    )


if __name__ == "__main__":
    main()
