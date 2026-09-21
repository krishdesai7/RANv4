# Evaluation & Metrics

RAN evaluates the quality of reweighted distributions using three complementary distance metrics, comparing simulation against data before and after reweighting.

---

## The Three Distance Metrics

All metrics are evaluated across every feature dimension on the held-out test split:

### 1. 1D Wasserstein-1 Distance

The earth mover's distance between the cumulative distribution functions $F_{\text{ref}}$ and $F_{\text{comp}}$:

$$\mathcal{W}_1 = \int |F_{\text{ref}}(t) - F_{\text{comp}}(t)| \, dt$$

In RAN's implementation, the two CDFs are **never accumulated separately**. Accumulating them separately causes catastrophic cancellation when subtracting two numbers near 1. Instead, signed weights are accumulated in a single scan, keeping the running value at the size of the answer and preserving float32 precision.

### 2. Jensen-Shannon Divergence

The symmetrized, bounded version of the Kullback-Leibler divergence:

$$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$

where $M = \frac{1}{2}(P + Q)$. Computed across uniform bins over the combined feature range.

### 3. Triangular Discriminator

A symmetric $f$-divergence with desirable numerical properties near zero:

$$\Delta(P, Q) = \int \frac{(p(x) - q(x))^2}{p(x) + q(x)} \, dx$$

---

## On-Device Vectorized Evaluation

Every metric evaluation in `deconvolve.evaluation.evaluate` runs directly on the JAX accelerator device:

- **Vectorized over dimensions**: A single JAX dispatch computes the metrics across all dimensions simultaneously.
- **Minimal host transfer**: Only the final scalar distance values cross back from the device to the host.
- **Performance**: Evaluates 100k-vs-100k samples in 6 dimensions in ~0.28 seconds.

---

## Output: `metrics.json`

Running `deconvolve evaluate` writes a structured `metrics.json` file inside the run directory:

```json
{
  "unweighted": {
    "wasserstein_1d": [0.354, 0.289],
    "js_divergence": [0.048, 0.039],
    "triangular": [0.092, 0.075]
  },
  "reweighted": {
    "wasserstein_1d": [0.012, 0.009],
    "js_divergence": [0.0008, 0.0006],
    "triangular": [0.0015, 0.0011]
  }
}
```
