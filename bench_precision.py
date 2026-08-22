"""The actual claim: does float32 change the UNFOLDING ACCURACY?

Same data, same seeds, same architecture -- only the dtype differs. Compares
the metric the project is judged on: Wasserstein(truth, reweighted gen).
"""
import os, sys
DTYPE = sys.argv[1]                      # "float64" | "float32"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = "1" if DTYPE == "float64" else "0"

import numpy as np, ran  # noqa: F401
import ran.models as models

# Repoint the model builders at the dtype under test.
_src = models.__dict__
for fn in ("build_generator", "build_discriminator"):
    assert fn in _src, sorted(k for k in _src if k.startswith("build"))

import re, pathlib
# models.py hardcodes "float64"; patch the module source in memory instead of
# editing the repo, so this probe is non-destructive.
src = pathlib.Path("src/ran/models.py").read_text().replace('"float64"', f'"{DTYPE}"')
ns: dict = {}
exec(compile(src, "ran/models.py", "exec"), ns)          # noqa: S102
models.build_generator = ns["build_generator"]
models.build_discriminator = ns["build_discriminator"]

import ran.train as T
T.build_generator = ns["build_generator"]
T.build_discriminator = ns["build_discriminator"]

from ran.data import RANDataset
from ran.rantypes import Events, Populations
from scipy.stats import wasserstein_distance

scalar = np.float64 if DTYPE == "float64" else np.float32
N, D, EPOCHS = 200_000, 6, 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = np.random.default_rng(0)
z_true = rng.normal(size=(N, D)); z_gen = rng.normal(loc=0.5, size=(N, D))
pops = Populations(
    mc=Events(z=z_gen, x=z_gen + 0.5 * rng.normal(size=(N, D))),
    data=z_true + 0.5 * rng.normal(size=(N, D)), truth=z_true,
).astype(scalar)

splits = RANDataset(batch_size=1024, seed=0, dtype=scalar).splits_from_data(pops.interleave())
result = T.train(splits, dim=D, hidden_units=64, n_layers=2, patience=99,
                 n_epochs=EPOCHS, seed=SEED)

raw = np.asarray(result.g(pops.mc.z)).ravel().astype(np.float64)
w = raw * len(raw) / raw.sum()
print(f"\n=== {DTYPE} (JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}) ===")
print(f"final val loss : {result.history['val_d'][-1]:.10f}")
print(f"epochs run     : {len(result.history['val_d'])}")
for i in range(D):
    before = wasserstein_distance(pops.truth[:, i].astype(np.float64), pops.mc.z[:, i].astype(np.float64))
    after = wasserstein_distance(pops.truth[:, i].astype(np.float64), pops.mc.z[:, i].astype(np.float64), v_weights=w)
    print(f"dim {i}: WD before={before:.8f}  after={after:.8f}  improvement={100*(before-after)/before:+.4f}%")

# --- ensemble mode -------------------------------------------------------
# n=1 cannot separate a precision penalty from adversarial run-to-run variance.
# Run:  for s in 0 1 2 3 4 5 6 7 8 9; do
#           uv run python bench_precision.py float64 $s >> f64.log
#           uv run python bench_precision.py float32 $s >> f32.log
#       done
# then compare the two distributions of mean improvement, not two single runs.
