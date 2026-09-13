# Tech Stack

- Python >= 3.14, managed with `uv` (no pip). Not 3.13: `src/ran/instrumentation/timing.py`
  uses PEP 758's unparenthesized `except OSError, ValueError:`, which is a
  `SyntaxError` on anything earlier — and ruff's formatter canonicalises the
  parenthesized form to it, so it will come back if someone "fixes" it
- Keras 3 on the **JAX** backend for training; `jax[cuda13]` on x86_64 Linux
- Typer for the CLI, Rich for logging and metrics tables
- Matplotlib for publication-quality plots; `pdflatex` (TeX Live, with
  siunitx, booktabs and pdflscape) for `ran report`, and only for that
- scipy for evaluation metrics (Wasserstein distance, Jensen-Shannon divergence)
- jaxtyping + beartype for shape/dtype checking on the training loop's array seams
- ruff (lint + format), pyrefly (types, `--min-severity info`), complexipy (max 10)
- TensorFlow is not a dependency, direct or transitive. JAX is the only array
  backend in the build, so nothing here has to negotiate for the GPU.

## Backend

JAX is the only backend in the build; there is no second framework competing
for the Keras backend slot or for the GPU.

`src/ran/__init__.py` sets `KERAS_BACKEND=jax` and `JAX_ENABLE_X64=0`. Keras 3
defaults to TensorFlow when that variable is unset, and TensorFlow is not
installed, so the pin makes `import keras` work at all. It must land before
the first keras import, which is why it lives in the package `__init__`;
`src/ran/training/train.py` keeps a cheap guard that raises a readable error if someone
sets `KERAS_BACKEND` to something else by hand.

## Gaussian Config Format

YAML files in `params/` use keys: `mu_gen`, `mu_true`, `sigma_gen`,
`sigma_true`, `sigma_detector`. Sigma values are promoted via
`sigma_to_covariance`: scalar → σ²I, vector → diag(σ²), matrix → used as-is.
