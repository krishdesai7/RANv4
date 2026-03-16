# Correlated Gaussian Dataset Support

## Summary

Replace the existing independent-dimension Gaussian dataset generator with a
fully general multivariate Gaussian generator. Users supply particle-level and
detector-level distribution parameters (means, covariances) via a YAML config
file. This is a clean break — the old `--dim`/`--smearing` CLI interface is
removed. Bare `uv run -m ran` without `--config` will error for Gaussian mode.

## YAML Config Format

Five required keys. `dim` is inferred from `len(mu_mc)`.

```yaml
mu_mc: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_mc: # (dim,) for diagonal or (dim,dim) for full covariance
  - [1.0, -0.54]
  - [-0.54, 2.25]
sigma_true:
  - [0.81, -0.702]
  - [-0.702, 1.69]
sigma_detector: [0.5, 0.8] # also supports (dim,dim) for correlated detector response
```

### Sigma interpretation

- **scalar**: `sigma_mc: 1.0` → isotropic covariance `1.0**2 * I`.
- **`(dim,)` vector**: per-dimension standard deviations; covariance
  matrix is `diag(sigma**2)`.
- **`(dim, dim)` matrix**: used as-is as the full covariance matrix.

This applies to all three sigma keys — `sigma_mc`, `sigma_true`, and
`sigma_detector`. A `(dim, dim)` `sigma_detector` models correlated detector smearing.

### Validation

- All five keys must be present.
- `mu_mc` and `mu_true` must be `(dim,)` vectors.
- Each `sigma_*` must be a scalar, `(dim,)`, or `(dim, dim)`, consistent with `dim`.
- Covariance matrices are validated by `scipy.linalg.cholesky`, which raises
  `LinAlgError` if the matrix is not symmetric or positive-definite. The Cholesky
  factor is computed once and reused for generation.

## Data Generation

### Particle-level draws

```python
z_true = rng.multivariate_normal(mu_true, cov_true, size=n_samples,
                                  check_valid='raise', method='svd')
z_gen  = rng.multivariate_normal(mu_mc,   cov_mc,   size=n_samples,
                                  check_valid='raise', method='svd')
```

### Detector-level smearing

Per-event smearing centered on each event's particle-level value, preserving
the `(z, x)` pairing. Vectorized via Cholesky decomposition:

```python
L = scipy.linalg.cholesky(cov_det, lower=True)  # computed once
s_data = rng.standard_normal(size=z_true.shape)
x_data = z_true + s_data @ L.T

s_sim = rng.standard_normal(size=z_gen.shape)
x_sim = z_gen + s_sim @ L.T
```

This is mathematically equivalent to drawing `x_i ~ N(z_i, Sigma_det)` for
each event, but fully vectorized.

**Critical**: the smearing is `x ~ N(z, Sigma_det)`, NOT `x = z + N(0, Sigma_det)`.
These are identical in this Cholesky formulation, but the conceptual model is
that detector response is conditioned on the true particle-level value.

## Changes to `datasets.py`

### `generate_gaussian_dataset` dual interface

The method accepts either a YAML file path or pre-parsed parameter dicts. This
is needed because `--load_run` and `evaluate.py` reconstruct data from stored
`config.json` arrays (no YAML file available).

```python
def generate_gaussian_dataset(self,
    config_path: str | Path | None = None,
    params: dict | None = None,
    n_samples: int = 10**6,
) -> DatasetSplits:
```

Exactly one of `config_path` or `params` must be provided. `params` is a dict
with the five keys (`mu_mc`, `mu_true`, `sigma_mc`, `sigma_true`, `sigma_detector`)
as numpy arrays or nested lists. The YAML path is just a convenience that reads
and parses into the same `params` dict.

A `parse_gaussian_config(config_path) -> dict` helper reads and validates the
YAML, returning the params dict. This is also used by `__main__.py` to obtain
params for storage in `config.json`.

### Removed parameters

- `smearing: float` — replaced by `sigma_detector` in config.
- `dim: int` — inferred from `mu_mc`.

### `_cache_key` / `_cache_path` updates

Cache key computed from a canonical JSON serialization of the five parsed
parameter arrays (as sorted nested lists), plus `n_samples` and `self.seed`.
This is deterministic regardless of whether the entry point was a YAML file or
stored `config.json` arrays. The old `smearing`/`dim`/`test_fraction` parameters
are removed from the key. (`test_fraction` was included before but is irrelevant
since the cache stores pre-split data.)

## Changes to `__main__.py`

### CLI interface

```zsh
uv run -m ran --config params/2d_corr.yaml
uv run -m ran --config params/2d_corr.yaml --n_samples 1000000
uv run -m ran --dataset jets --variables m,M,w
uv run -m ran --load_run runs/2026-03-16T...
```

- Remove `smearing`, `dim` flags.
- Add `config` flag (path to YAML). Required when `dataset == "gaussian"`.
  Running without `--config` in Gaussian mode raises `ValueError`.
- `dim` is inferred internally, still passed to model builders.

### `config.json` per run

The saved `config.json` includes the full parsed Gaussian parameters (all five
arrays serialized as nested lists) so that runs are self-contained and
reloadable without the original YAML file. On `--load_run`, these stored arrays
are passed directly via the `params` kwarg.

## Changes to `evaluate.py`

`_load_splits` updated to reconstruct the dataset from the stored parameter
arrays in `config.json`, passing them via the `params` kwarg to
`generate_gaussian_dataset`.

## No changes required

- `models.py` — already takes `dim` as input shape.
- `train.py` — already dimension-agnostic.
- `plotting.py` — iterates over `dim` for per-dimension 1D marginals.
- `evaluate.py` metric computation — already per-dimension.

## New dependency

- `pyyaml` — added to `pyproject.toml` for YAML parsing.
