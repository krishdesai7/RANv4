# Correlated Gaussian Dataset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the independent-dimension Gaussian generator with a fully general multivariate Gaussian generator configured via YAML.

**Architecture:** A `parse_gaussian_config()` helper reads YAML and promotes sigma values (scalar → `σ²I`, vector → `diag(σ²)`, matrix → as-is) into covariance matrices. `generate_gaussian_dataset()` accepts either a config path or pre-parsed params dict, draws particle-level samples with `rng.multivariate_normal`, and applies detector smearing via Cholesky decomposition. The CLI replaces `--dim`/`--smearing` with `--config`.

**Tech Stack:** Python 3.13+, NumPy, SciPy (`linalg.cholesky`), PyYAML, pytest (new dev dependency), TensorFlow/Keras (unchanged)

**Spec:** `docs/superpowers/specs/2026-03-16-correlated-gaussians-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `ran/data/config.py` | YAML parsing, sigma promotion, validation |
| Modify | `ran/data/datasets.py` | New `generate_gaussian_dataset` signature, multivariate generation |
| Modify | `ran/data/__init__.py` | Export `parse_gaussian_config` |
| Modify | `ran/__main__.py` | Replace `--dim`/`--smearing` with `--config`, store params in `config.json` |
| Modify | `ran/evaluate.py` | `_load_splits` uses `params` kwarg |
| Modify | `pyproject.toml` | Add `pyyaml` dependency, `pytest` dev dependency |
| Create | `tests/test_config.py` | Tests for YAML parsing and sigma promotion |
| Create | `tests/test_datasets.py` | Tests for multivariate dataset generation |
| Create | `params/1d_default.yaml` | Example: 1D uncorrelated config |
| Create | `params/2d_correlated.yaml` | Example: 2D correlated config |

---

## Chunk 1: Config Parsing

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml and pytest**

Add `pyyaml` to `[project.dependencies]` and `pytest` to `[dependency-groups.dev]` in `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --group dev`
Expected: clean install, no errors

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add pyyaml and pytest dependencies"
```

---

### Task 2: Write config parser tests

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Write tests for sigma promotion**

`tests/test_config.py`:

```python
import numpy as np
import pytest
from ran.data.config import parse_gaussian_config, sigma_to_covariance

class TestSigmaToCovariance:
    """Test the three sigma forms: scalar, vector, matrix."""

    def test_scalar_1d(self):
        cov = sigma_to_covariance(2.0, dim=1)
        expected = np.array([[4.0]])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_scalar_3d(self):
        cov = sigma_to_covariance(1.5, dim=3)
        expected = 2.25 * np.eye(3)
        np.testing.assert_array_almost_equal(cov, expected)

    def test_vector(self):
        cov = sigma_to_covariance([1.0, 2.0], dim=2)
        expected = np.diag([1.0, 4.0])
        np.testing.assert_array_almost_equal(cov, expected)

    def test_matrix_passthrough(self):
        mat = [[1.0, 0.5], [0.5, 2.0]]
        cov = sigma_to_covariance(mat, dim=2)
        np.testing.assert_array_almost_equal(cov, mat)

    def test_vector_wrong_dim(self):
        with pytest.raises(ValueError, match="dim"):
            sigma_to_covariance([1.0, 2.0, 3.0], dim=2)

    def test_matrix_wrong_shape(self):
        with pytest.raises(ValueError, match="dim"):
            sigma_to_covariance([[1.0, 0.0], [0.0, 1.0]], dim=3)

    def test_not_positive_definite(self):
        """A matrix with negative eigenvalue should fail."""
        bad = [[1.0, 5.0], [5.0, 1.0]]
        with pytest.raises(np.linalg.LinAlgError):
            sigma_to_covariance(bad, dim=2)

    def test_asymmetric_matrix_raises(self):
        """An asymmetric matrix should be rejected."""
        asym = [[1.0, 0.5], [999.0, 2.0]]
        with pytest.raises(ValueError, match="symmetric"):
            sigma_to_covariance(asym, dim=2)

    def test_negative_scalar_raises(self):
        """Negative scalar sigma is physically nonsensical."""
        with pytest.raises(ValueError, match="negative"):
            sigma_to_covariance(-1.0, dim=2)

    def test_negative_vector_element_raises(self):
        """Negative elements in sigma vector should be rejected."""
        with pytest.raises(ValueError, match="negative"):
            sigma_to_covariance([1.0, -0.5], dim=2)
```

- [ ] **Step 2: Write tests for YAML parsing**

Append to `tests/test_config.py`:

```python
from pathlib import Path
import tempfile
import yaml

class TestParseGaussianConfig:
    """Test full YAML config parsing."""

    def _write_yaml(self, data: dict, tmp_path: Path) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        return p

    def test_valid_2d_config(self, tmp_path):
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = self._write_yaml(cfg, tmp_path)
        params = parse_gaussian_config(path)
        assert params["dim"] == 2
        assert params["mu_mc"].shape == (2,)
        assert params["cov_mc"].shape == (2, 2)
        assert params["cov_true"].shape == (2, 2)
        assert params["cov_detector"].shape == (2, 2)

    def test_scalar_sigma(self, tmp_path):
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        params = parse_gaussian_config(path)
        assert params["dim"] == 1
        np.testing.assert_array_almost_equal(params["cov_mc"], [[1.0]])
        np.testing.assert_array_almost_equal(params["cov_detector"], [[0.25]])

    def test_missing_key(self, tmp_path):
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            # missing sigma_true and sigma_detector
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="missing"):
            parse_gaussian_config(path)

    def test_dim_mismatch(self, tmp_path):
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.5],  # dim 1 vs dim 2
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = self._write_yaml(cfg, tmp_path)
        with pytest.raises(ValueError, match="dim"):
            parse_gaussian_config(path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ran.data.config'`

- [ ] **Step 4: Commit**

```bash
git add tests/test_config.py
git commit -m "Add tests for Gaussian config parsing"
```

---

### Task 3: Implement config parser

**Files:**
- Create: `ran/data/config.py`
- Modify: `ran/data/__init__.py`

- [ ] **Step 1: Implement `sigma_to_covariance` and `parse_gaussian_config`**

`ran/data/config.py`:

```python
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

import yaml


def sigma_to_covariance(
    sigma: float | list | npt.NDArray,
    dim: int,
) -> npt.NDArray[np.double]:
    """Promote sigma (scalar, vector, or matrix) to a (dim, dim) covariance matrix.

    - scalar: σ²I
    - (dim,) vector: diag(σ²)
    - (dim, dim) matrix: used as-is

    Validates positive-definiteness via Cholesky decomposition.
    """
    arr: npt.NDArray[np.double] = np.atleast_1d(np.asarray(sigma, dtype=np.double))

    cov: npt.NDArray[np.double]
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        # Scalar
        val = float(arr.ravel()[0])
        if val < 0:
            raise ValueError(f"sigma scalar must be non-negative, got {val}")
        cov = val ** 2 * np.eye(dim, dtype=np.double)
    elif arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(
                f"sigma vector has length {arr.shape[0]}, expected dim={dim}"
            )
        if np.any(arr < 0):
            raise ValueError("sigma vector elements must be non-negative")
        cov = np.diag(arr ** 2).astype(np.double)
    elif arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(
                f"sigma matrix has shape {arr.shape}, expected ({dim}, {dim})"
            )
        if not np.allclose(arr, arr.T):
            raise ValueError("sigma matrix must be symmetric")
        cov = arr
    else:
        raise ValueError(f"sigma must be scalar, 1D, or 2D, got ndim={arr.ndim}")

    # Validate positive-definite via scipy cholesky
    cholesky(cov, lower=True)
    return cov


REQUIRED_KEYS: set[str] = {"mu_mc", "mu_true", "sigma_mc", "sigma_true", "sigma_detector"}


def parse_gaussian_config(config_path: str | Path) -> dict:
    """Parse a Gaussian YAML config file.

    Returns a dict with keys:
        dim (int), mu_mc, mu_true (1D arrays),
        cov_mc, cov_true, cov_detector (2D covariance matrices).
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        raw: dict = yaml.safe_load(f)

    missing: set[str] = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    mu_mc: npt.NDArray[np.double] = np.asarray(raw["mu_mc"], dtype=np.double).ravel()
    mu_true: npt.NDArray[np.double] = np.asarray(raw["mu_true"], dtype=np.double).ravel()

    dim: int = mu_mc.shape[0]
    if mu_true.shape[0] != dim:
        raise ValueError(
            f"mu_true has dim {mu_true.shape[0]}, expected dim={dim} (from mu_mc)"
        )

    cov_mc: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_mc"], dim)
    cov_true: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_true"], dim)
    cov_detector: npt.NDArray[np.double] = sigma_to_covariance(raw["sigma_detector"], dim)

    return {
        "dim": dim,
        "mu_mc": mu_mc,
        "mu_true": mu_true,
        "cov_mc": cov_mc,
        "cov_true": cov_true,
        "cov_detector": cov_detector,
    }
```

- [ ] **Step 2: Export from `ran/data/__init__.py`**

Read `ran/data/__init__.py` and add `parse_gaussian_config` to its imports. Add:

```python
from ran.data.config import parse_gaussian_config
```

- [ ] **Step 3: Run tests**

Run: `uv run --group dev pytest tests/test_config.py -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add ran/data/config.py ran/data/__init__.py
git commit -m "Implement Gaussian config parser with sigma promotion"
```

---

## Chunk 2: Dataset Generation

### Task 4: Write dataset generation tests

**Files:**
- Create: `tests/test_datasets.py`

- [ ] **Step 1: Write tests for multivariate generation**

`tests/test_datasets.py`:

```python
import numpy as np
import pytest
import tempfile
from pathlib import Path

import yaml

from ran.data.datasets import RAN_Dataset


def _write_config(params: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(params))
    return p


class TestGenerateGaussianDataset:
    """Test multivariate Gaussian dataset generation."""

    def test_1d_uncorrelated(self, tmp_path):
        """1D scalar sigma should produce valid splits."""
        cfg = {
            "mu_mc": [0.5],
            "mu_true": [0.0],
            "sigma_mc": 0.9,
            "sigma_true": 1.0,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=1000)
        # Verify we get all three splits
        assert splits.train is not None
        assert splits.val is not None
        assert splits.test is not None

    def test_2d_correlated_shapes(self, tmp_path):
        """2D with full covariance should produce correct shapes."""
        cfg = {
            "mu_mc": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_mc": [[1.0, -0.54], [-0.54, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=2000)
        # Check shapes by iterating test set
        for features, y in splits.test:
            assert features["z"].shape[1] == 2
            assert features["x"].shape[1] == 2
            break

    def test_params_dict_interface(self, tmp_path):
        """Passing params dict directly should work (for --load_run)."""
        params = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        ds = RAN_Dataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(params=params, n_samples=1000)
        assert splits.train is not None

    def test_both_config_and_params_raises(self, tmp_path):
        """Providing both config_path and params should error."""
        cfg = {"mu_mc": [0.0], "mu_true": [0.5], "sigma_mc": 1.0,
               "sigma_true": 0.9, "sigma_detector": 0.5}
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(config_path=path, params=cfg, n_samples=100)

    def test_neither_config_nor_params_raises(self):
        """Providing neither config_path nor params should error."""
        ds = RAN_Dataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            ds.generate_gaussian_dataset(n_samples=100)

    def test_caching(self, tmp_path):
        """Second call with same config should hit cache."""
        cfg = {
            "mu_mc": [0.0],
            "mu_true": [0.5],
            "sigma_mc": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"
        ds = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds.generate_gaussian_dataset(config_path=path, n_samples=500)
        # Cache file should exist
        cache_files = list(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files) == 1
        # Second call should load from cache (no error)
        ds2 = RAN_Dataset(batch_size=64, seed=42, cache_dir=cache_dir)
        ds2.generate_gaussian_dataset(config_path=path, n_samples=500)

    def test_smearing_preserves_event_coupling(self, tmp_path):
        """Detector-level values should be correlated with particle-level."""
        cfg = {
            "mu_mc": [0.0, 0.0],
            "mu_true": [0.0, 0.0],
            "sigma_mc": [1.0, 1.0],
            "sigma_true": [1.0, 1.0],
            "sigma_detector": [0.1, 0.1],  # tight smearing
        }
        path = _write_config(cfg, tmp_path)
        ds = RAN_Dataset(batch_size=10000, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=10000)
        # With tiny smearing, x ≈ z
        for features, y in splits.test:
            z = features["z"].numpy()
            x = features["x"].numpy()
            for d in range(2):
                corr = np.corrcoef(z[:, d], x[:, d])[0, 1]
                assert corr > 0.95, f"dim {d}: corr={corr}, expected >0.95"
            break
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_datasets.py -v`
Expected: FAIL — `generate_gaussian_dataset() got an unexpected keyword argument 'config_path'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_datasets.py
git commit -m "Add tests for multivariate Gaussian dataset generation"
```

---

### Task 5: Implement multivariate dataset generation

**Files:**
- Modify: `ran/data/datasets.py`

- [ ] **Step 1: Rewrite `generate_gaussian_dataset` and update cache methods**

Replace the existing `_cache_key`, `_cache_path`, and `generate_gaussian_dataset` methods in `ran/data/datasets.py`. The new implementation:

- `_cache_key` takes the `parsed` dict (with `cov_*` keys, i.e. post-promotion covariance matrices) and `n_samples`. This ensures the same physical config always produces the same cache key regardless of entry point (YAML vs params dict).
- `_cache_path` takes parsed dict and n_samples
- `generate_gaussian_dataset` accepts `config_path` or `params`, uses `parse_gaussian_config` for YAML, calls `rng.multivariate_normal` for particle-level draws and Cholesky smearing for detector-level

```python
# New imports needed at top of file:
from scipy.linalg import cholesky
from ran.data.config import parse_gaussian_config

# Replace _cache_key:
def _cache_key(self, parsed: dict, n_samples: int) -> str:
    """Hash the promoted covariance matrices for a canonical cache key."""
    key_data = {
        "mu_mc": parsed["mu_mc"].tolist(),
        "mu_true": parsed["mu_true"].tolist(),
        "cov_mc": parsed["cov_mc"].tolist(),
        "cov_true": parsed["cov_true"].tolist(),
        "cov_detector": parsed["cov_detector"].tolist(),
        "n_samples": n_samples,
        "seed": self.seed,
    }
    return hashlib.sha256(
        json.dumps(key_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

# Replace _cache_path:
def _cache_path(self, parsed: dict, n_samples: int) -> Path:
    cache_key = self._cache_key(parsed, n_samples)
    return self.cache_dir / f"gaussian_{cache_key}.npz"

# Replace generate_gaussian_dataset:
def generate_gaussian_dataset(self,
    config_path: str | Path | None = None,
    params: dict | None = None,
    n_samples: int = 10 ** 6,
) -> DatasetSplits:
    if (config_path is None) == (params is None):
        raise ValueError(
            "Exactly one of config_path or params must be provided"
        )

    if config_path is not None:
        parsed = parse_gaussian_config(config_path)
    else:
        from ran.data.config import sigma_to_covariance
        mu_mc = np.asarray(params["mu_mc"], dtype=np.double).ravel()
        mu_true = np.asarray(params["mu_true"], dtype=np.double).ravel()
        dim = mu_mc.shape[0]
        if mu_true.shape[0] != dim:
            raise ValueError(
                f"mu_true has dim {mu_true.shape[0]}, expected {dim}"
            )
        parsed = {
            "dim": dim,
            "mu_mc": mu_mc,
            "mu_true": mu_true,
            "cov_mc": sigma_to_covariance(params["sigma_mc"], dim),
            "cov_true": sigma_to_covariance(params["sigma_true"], dim),
            "cov_detector": sigma_to_covariance(params["sigma_detector"], dim),
        }

    dim: int = parsed["dim"]
    mu_mc: npt.NDArray[np.double] = parsed["mu_mc"]
    mu_true: npt.NDArray[np.double] = parsed["mu_true"]
    cov_mc: npt.NDArray[np.double] = parsed["cov_mc"]
    cov_true: npt.NDArray[np.double] = parsed["cov_true"]
    cov_detector: npt.NDArray[np.double] = parsed["cov_detector"]

    cache_path: Path = self._cache_path(parsed, n_samples)
    self.cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading dataset from cache: {cache_path}")
        with np.load(cache_path) as data:
            z = data["z"]
            x = data["x"]
            y = data["y"]
    else:
        rng = np.random.default_rng(self.seed)

        z_true = rng.multivariate_normal(
            mu_true, cov_true, size=n_samples,
            check_valid='raise', method='svd',
        )
        z_gen = rng.multivariate_normal(
            mu_mc, cov_mc, size=n_samples,
            check_valid='raise', method='svd',
        )

        L_det = cholesky(cov_detector, lower=True)

        s_data = rng.standard_normal(size=z_true.shape)
        x_data = z_true + s_data @ L_det.T

        s_sim = rng.standard_normal(size=z_gen.shape)
        x_sim = z_gen + s_sim @ L_det.T

        y_nat = np.ones(n_samples, dtype=np.ubyte)
        y_MC = np.zeros(n_samples, dtype=np.ubyte)

        z = np.concatenate((z_true, z_gen), axis=0)
        x = np.concatenate((x_data, x_sim), axis=0)
        y = np.concatenate((y_nat, y_MC), axis=0)

        np.savez_compressed(cache_path, z=z, x=x, y=y)
        print(f"Generated and saved dataset to cache: {cache_path}")

    self.dataset = self._build_dataset(z, x, y)
    self.splits = self._split_dataset(self.dataset)
    return self.splits
```

- [ ] **Step 2: Run tests**

Run: `uv run --group dev pytest tests/test_datasets.py tests/test_config.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add ran/data/datasets.py
git commit -m "Implement multivariate Gaussian generation with Cholesky smearing"
```

---

## Chunk 3: CLI and Evaluation Updates

### Task 6: Update `__main__.py`

**Files:**
- Modify: `ran/__main__.py`

- [ ] **Step 1: Replace CLI interface**

Update `main()` in `ran/__main__.py`:

- Remove `smearing`, `dim` parameters
- Add `config: str | None = None` parameter
- For Gaussian mode: call `parse_gaussian_config(config)` to get params, infer `dim`
- Store full params in `config.json` (as nested lists)
- On `--load_run` for Gaussian: pass stored params via `params` kwarg

Key changes to the function signature:

```python
def main(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    config: str | None = None,
    dataset: str = "gaussian",
    variables: tuple[str, ...] = ("m", "M", "w", "tau21", "zg", "sdm"),
    load_run: str | None = None,
) -> None:
```

In the Gaussian branch:

```python
if dataset == "gaussian":
    if load_run is not None:
        gaussian_params = config_data.get("gaussian_params")
        dim = gaussian_params["dim"]
        # Strip dim before passing — it's metadata, not a config key
        raw_params = {k: v for k, v in gaussian_params.items() if k != "dim"}
        splits = RAN_Dataset(
            batch_size=batch_size
        ).generate_gaussian_dataset(
            params=raw_params,
            n_samples=n_samples,
        )
    else:
        if config is None:
            raise ValueError(
                "Gaussian mode requires --config path/to/config.yaml"
            )
        from ran.data.config import parse_gaussian_config
        gaussian_params = parse_gaussian_config(config)
        dim = gaussian_params["dim"]
        splits = RAN_Dataset(
            batch_size=batch_size
        ).generate_gaussian_dataset(
            config_path=config,
            n_samples=n_samples,
        )
```

When saving `config.json`, include `gaussian_params` with arrays as nested lists:

```python
config_out = {"batch_size": batch_size, "n_samples": n_samples,
              "dim": dim, "dataset": dataset}
if dataset == "gaussian":
    config_out["gaussian_params"] = {
        "dim": dim,
        "mu_mc": gaussian_params["mu_mc"].tolist()
                 if hasattr(gaussian_params["mu_mc"], "tolist")
                 else gaussian_params["mu_mc"],
        "mu_true": gaussian_params["mu_true"].tolist()
                   if hasattr(gaussian_params["mu_true"], "tolist")
                   else gaussian_params["mu_true"],
        "sigma_mc": gaussian_params["cov_mc"].tolist()
                    if hasattr(gaussian_params["cov_mc"], "tolist")
                    else gaussian_params["cov_mc"],
        "sigma_true": gaussian_params["cov_true"].tolist()
                      if hasattr(gaussian_params["cov_true"], "tolist")
                      else gaussian_params["cov_true"],
        "sigma_detector": gaussian_params["cov_detector"].tolist()
                          if hasattr(gaussian_params["cov_detector"], "tolist")
                          else gaussian_params["cov_detector"],
    }
else:
    config_out["variables"] = list(variables)
```

- [ ] **Step 2: Verify Gaussian mode runs**

Create a test config and run:

```bash
cat > /tmp/test_1d.yaml << 'EOF'
mu_mc: [0.5]
mu_true: [0.0]
sigma_mc: 0.9
sigma_true: 1.0
sigma_detector: 0.5
EOF
uv run -m ran --config /tmp/test_1d.yaml --n_samples 10000
```

Expected: trains, produces plots and `config.json` in `runs/` directory

- [ ] **Step 3: Verify jet mode still works**

Run: `uv run -m ran --dataset jets --n_samples 10000`
Expected: trains without errors (jet path is unchanged)

- [ ] **Step 4: Commit**

```bash
git add ran/__main__.py
git commit -m "Replace --dim/--smearing CLI with --config for Gaussian mode"
```

---

### Task 7: Update `evaluate.py`

**Files:**
- Modify: `ran/evaluate.py`

- [ ] **Step 1: Update `_load_splits` for new Gaussian interface**

In `ran/evaluate.py`, update the `_load_splits` function. The Gaussian branch
should read `gaussian_params` from the config and pass the raw sigma values
via the `params` kwarg:

```python
if dataset == "gaussian":
    gaussian_params = config["gaussian_params"]
    raw_params = {k: v for k, v in gaussian_params.items() if k != "dim"}
    return RAN_Dataset(batch_size=batch_size).generate_gaussian_dataset(
        params=raw_params, n_samples=n_samples,
    )
```

- [ ] **Step 2: Verify evaluation works on a saved run**

Run against the run created in Task 6:

```bash
uv run -m ran.evaluate --run_dir=runs/<latest-run-dir> --force
```

Expected: prints Wasserstein and JS metrics without errors

- [ ] **Step 3: Commit**

```bash
git add ran/evaluate.py
git commit -m "Update evaluate.py to use params dict for Gaussian dataset reconstruction"
```

---

### Task 8: Create example YAML configs and update CLAUDE.md

**Files:**
- Create: `params/1d_default.yaml`
- Create: `params/2d_correlated.yaml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create example configs**

`params/1d_default.yaml` (reproduces the old default behavior):

```yaml
mu_mc: [0.5]
mu_true: [0.0]
sigma_mc: 0.9
sigma_true: 1.0
sigma_detector: 0.5
```

`params/2d_correlated.yaml` (inspired by gaussian_parameters.tex):

```yaml
mu_mc: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_mc:
  - [1.0, -0.9]
  - [-0.9, 2.25]
sigma_true:
  - [0.81, -0.702]
  - [-0.702, 1.69]
sigma_detector: [0.5, 0.8]
```

- [ ] **Step 2: Update CLAUDE.md running instructions**

Replace the Gaussian running examples in `CLAUDE.md`:

```markdown
uv run -m ran --config params/1d_default.yaml              # 1D uncorrelated
uv run -m ran --config params/2d_correlated.yaml           # 2D with covariance
uv run -m ran --dataset jets                               # train on all 6 jet variables
uv run -m ran --load_run=runs/2026-03-14T061023Z           # reload a saved run
```

- [ ] **Step 3: Commit**

```bash
git add params/ CLAUDE.md
git commit -m "Add example YAML configs and update running instructions"
```

---

### Task 9: End-to-end smoke test

- [ ] **Step 1: Run 2D correlated training end-to-end**

```bash
uv run -m ran --config params/2d_correlated.yaml --n_samples 50000
```

Expected: trains to completion, saves plots (detector_level.pdf, particle_level.pdf, losses.pdf) and config.json

- [ ] **Step 2: Reload the run**

```bash
uv run -m ran --load_run runs/<the-run-dir-from-step-1>
```

Expected: loads model, regenerates plots without retraining

- [ ] **Step 3: Evaluate the run**

```bash
uv run -m ran.evaluate --run_dir=runs/<the-run-dir-from-step-1> --force
```

Expected: prints metrics table

- [ ] **Step 4: Run full test suite**

```bash
uv run --group dev pytest tests/ -v
```

Expected: all tests PASS
