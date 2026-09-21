# Precision & Hardware

RAN is strictly **float32 end-to-end**. This section explains how precision is enforced, how backend initialization is handled, and how hardware divergence is prevented.

---

## The Float32 Pin

High Energy Physics calculations often mix float64 and float32 unpredictably. RAN eliminates silent casts and memory bloat through a strict float32 design:

1. **`EVENT_DTYPE`**: Pinned in `deconvolve.coretypes.constants` as `np.float32`.
2. **`EventArray`**: Pinned in `deconvolve.coretypes.types` as `Float[Array, "N D"]` with `beartype` validation.
3. **`JAX_ENABLE_X64=0`**: Set before JAX is imported, preventing silent upcasting to float64.

---

## Backend Bootstrapping (`ran/__init__.py`)

Keras 3 and JAX inspect environment variables **once**, at the exact moment they are imported.

To ensure consistent behavior, `ran/__init__.py` sets the following defaults before any other submodule is loaded:

```python
import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "0")
```

If a user script imports `keras` or `jax` before importing `ran`, `deconvolve.training.engine` detects this and raises an informative `RuntimeError` rather than failing cryptically inside an XLA trace.

---

## Deterministic Matrix Multiplication (`HIGHEST`)

Modern accelerators (NVIDIA Ampere/Hopper) run matrix multiplications at reduced precision (TF32) by default, trading numerical accuracy for throughput.

For synthetic Gaussian sampling and covariance decomposition, this hardware-dependent precision causes the same seed to produce different random numbers on different machines (e.g. login node CPU vs compute node A100).

To make datasets hardware-invariant:

```python
jax.lax.Precision.HIGHEST
```

is explicitly set during sample generation and Cholesky smearing. A dataset drawn with seed 42 produces the exact same bitwise numbers on a MacBook CPU as on an NVIDIA A100 cluster node.

---

## Where Float64 Still Lives

Float64 is preserved in exactly two non-training locations:

1. **Host-Side Bin Edges**: `np.linspace` builds histogram bin edges on the host in float32 so device tracing never re-rounds them.
2. **Integrity Assertions**: Verifying that event counts match exact integers and that weights normalize to mean 1 accumulates in float64, because float32 ceases to represent consecutive integers beyond $2^{24} \approx 1.67 \times 10^7$.
