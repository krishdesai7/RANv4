# Timing & Profiling

Deconvolve includes a zero-dependency, low-overhead microsecond timing harness in `deconvolve.instrumentation.timing`.

---

## Phase Scopes

Execution phases are tracked using the `phase` context manager:

```python
from deconvolve.instrumentation import phase

with phase("train_epoch", epoch=1):
    train_step()
```

### Features

- **Nested Hierarchy**: Accurately tracks parent-child execution stages (e.g. `training` $\to$ `epoch` $\to$ `discriminator_step`).
- **Microsecond Precision**: Measures wall-clock duration via `time.perf_counter_ns()`.
- **Low Overhead**: Captures timing records in an in-memory ring buffer with negligible impact on JIT-compiled loop execution.
- **Run Artifact Output**: Serializes a complete execution profile to `timing.json` inside the run directory when enabled.

---

## Controlling Profiling

Profiling is disabled by default to avoid memory accumulation on long runs. Enable it via the environment:

```shell
export DECONVOLVE_TIMING=1
deconvolve train --config params/1d_default.yaml
```

When enabled, `timing.json` records per-phase statistics (calls, total duration, mean, min, max, standard deviation), which are rendered into the final LaTeX report dossier.
