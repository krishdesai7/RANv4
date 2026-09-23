# API Reference Overview

The Deconvolve Python library is structured into specialized modules. The following sections provide auto-generated reference documentation directly from the source code, type annotations, and docstrings.

---

## Subpackages

| Subpackage | Primary Responsibilities |
| :--- | :--- |
| **[`deconvolve.training`](training.md)** | Model architectures (`build_generator`, `build_discriminator`), JAX training engine (`train`), and MMD caching. |
| **[`deconvolve.data`](data.md)** | Dataset loading, Gaussian synthesis (`_draw_gaussian`), Zenodo jet caching, and on-device batching. |
| **[`deconvolve.evaluation`](evaluation.md)** | On-device vectorized distance metrics (Wasserstein, JS, triangular) and ratio plotting. |
| **[`deconvolve.baselines`](baselines.md)** | IBU (Iterative Bayesian Unfolding) and OmniFold comparison baselines. |
| **[`deconvolve.uncertainty`](uncertainty.md)** | Crossed ANOVA variance decomposition and bin-to-bin covariance estimation. |
| **[`deconvolve.coretypes`](coretypes.md)** | Shared event dataclasses (`Populations`, `Events`, `ZXY`), constants, and enums. |
| **[`deconvolve.instrumentation`](instrumentation.md)** | Execution timing harness (`phase`), logging configuration, and metrics collection. |

---

## Import Conventions

Public components should be imported directly from their respective submodules:

```python
from deconvolve.data import DatasetSplits, Populations
from deconvolve.evaluation import evaluate_runs
from deconvolve.coretypes import EventArray, Split
from deconvolve.training import build_generator, train
```
