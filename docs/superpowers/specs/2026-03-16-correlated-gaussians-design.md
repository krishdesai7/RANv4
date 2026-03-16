# Correlated Gaussian Dataset Support

## Summary

Replace the existing independent-dimension Gaussian dataset generator with a
fully general multivariate Gaussian generator. Users supply particle-level and
detector-level distribution parameters (means, covariances) via a YAML config
file. This is a clean break — the old `--dim`/`--smearing` CLI interface is
removed.

## YAML Config Format

Five required keys. `dim` is inferred from `len(mu_mc)`.

```yaml
mu_mc: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_mc:            # (dim,) for diagonal or (dim,dim) for full covariance
  - [1.0, -0.54]
  - [-0.54, 2.25]
sigma_true:
  - [0.81, -0.702]
  - [-0.702, 1.69]
sigma_det: [0.5, 0.8]
```

### Sigma interpretation

- `(dim,)` vector: treated as per-dimension standard deviations; covariance
  matrix is `diag(sigma**2)`.
- `(dim, dim)` matrix: used as-is as the full covariance matrix.

### Validation

- All five keys must be present.
- `mu_mc` and `mu_true` must be `(dim,)` vectors.
- Each `sigma_*` must be `(dim,)` or `(dim, dim)`, consistent with `dim`.
- Covariance matrices are validated by `scipy.linalg.cholesky`, which raises
  `LinAlgError` if the matrix is not symmetric positive-definite.

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
L = scipy.linalg.cholesky(cov_det, lower=True)
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

### `generate_gaussian_dataset` new signature

```python
def generate_gaussian_dataset(self,
    config_path: str | Path,
    n_samples: int = 10**6,
) -> DatasetSplits:
```

- Reads YAML config from `config_path`.
- Infers `dim` from `mu_mc`.
- Promotes `(dim,)` sigmas to `diag(sigma**2)`.
- Generates data as described above.
- Caching: cache key includes a hash of the full YAML content + `n_samples` +
  `seed`.

### Removed parameters

- `smearing: float` — replaced by `sigma_det` in YAML.
- `dim: int` — inferred from config.

### `_cache_key` / `_cache_path` updates

Cache key computed from a hash of the YAML content string, `n_samples`, and
`self.seed`. The `smearing`/`dim` parameters are removed from the key.

## Changes to `__main__.py`

### CLI interface

```
uv run -m ran --config params/2d_corr.yaml
uv run -m ran --config params/2d_corr.yaml --n_samples 1000000
uv run -m ran --dataset jets --variables m,M,w
uv run -m ran --load_run runs/2026-03-16T...
```

- Remove `smearing`, `dim` flags.
- Add `config` flag (path to YAML). Required when `dataset == "gaussian"`.
- `dim` is inferred internally, still passed to model builders.

### `config.json` per run

The saved `config.json` includes the full parsed Gaussian parameters (all five
arrays serialized as nested lists) so that runs are self-contained and
reloadable without the original YAML file.

## Changes to `evaluate.py`

`_load_splits` updated to reconstruct the dataset from the stored parameters in
`config.json` rather than from `smearing`/`dim`.

## No changes required

- `models.py` — already takes `dim` as input shape.
- `train.py` — already dimension-agnostic.
- `plotting.py` — iterates over `dim` for per-dimension 1D marginals.
- `evaluate.py` metric computation — already per-dimension.

## New dependency

- `pyyaml` — added to `pyproject.toml` for YAML parsing.
