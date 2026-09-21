# Configuration

RAN supports two dataset regimes: synthetic **Gaussian toy datasets** configured via YAML, and **Jet Substructure observables** pulled from Zenodo.

---

## Gaussian Datasets (YAML)

Gaussian toy datasets specify the mean and covariance for generation, nature (truth), and detector smearing.

### File Format

A Gaussian configuration file is a YAML document containing:

```yaml
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9          # scalar, vector, or full matrix
sigma_true: 1.0
sigma_detector: 0.5
```

### Sigma-to-Covariance Promotion

RAN automatically promotes `sigma_*` entries into `(dim, dim)` positive-definite covariance matrices:

- **Scalar** $\sigma \to \sigma^2 I$ (isotropic diagonal covariance)
- **Vector** $[\sigma_1, \sigma_2, \dots] \to \text{diag}(\sigma_1^2, \sigma_2^2, \dots)$ (uncorrelated diagonal covariance)
- **Matrix** $\Sigma \to$ used as-is (full correlated covariance)

Positive-definiteness is verified via Cholesky decomposition during configuration parsing.

### Multidimensional Examples

Pre-configured examples are provided in the `params/` directory:

=== "1D Uncorrelated (`params/1d_default.yaml`)"
    ```yaml
    mu_gen: [0.5]
    mu_true: [0.0]
    sigma_gen: 0.9
    sigma_true: 1.0
    sigma_detector: 0.5
    ```

=== "2D Correlated (`params/2d_correlated.yaml`)"
    ```yaml
    mu_gen: [0.5, -0.3]
    mu_true: [0.0, 0.0]
    sigma_gen:
      - [0.9, 0.2]
      - [0.2, 0.8]
    sigma_true:
      - [1.0, 0.0]
      - [0.0, 1.0]
    sigma_detector: 0.5
    ```

=== "4D / 6D Correlated"
    See `params/4d_correlated.yaml` and `params/6d_correlated.yaml` for higher-dimensional covariance structures.

---

## Jet Substructure Configuration

Jet substructure observables are selected directly via CLI flags or config dictionaries:

```shell
ran train -D jets -v m -v w
```

### Supported Observables

The dataset contains 12 jet observables from Pythia 8 and Herwig 7 simulations:

| Short Name | Observable | Description |
| :--- | :--- | :--- |
| `m` | $m$ | Jet invariant mass |
| `w` | $w$ | Jet width / girth |
| `mult` | $n_{\text{constituents}}$ | Constituent multiplicity |
| `ptd` | $p_T D$ | Dispersion of constituent $p_T$ |
| `sd_m` | $m_{sd}$ | Soft Drop groomed mass |
| `sd_z` | $z_g$ | Soft Drop momentum fraction |
| `sd_dr` | $\Delta R_g$ | Soft Drop opening angle |
| `tau1` - `tau4` | $\tau_1, \tau_2, \tau_3, \tau_4$ | N-subjettiness variables |
| `tau21` | $\tau_{21} = \tau_2 / \tau_1$ | Subjettiness ratio |

---

## Configuration Precedence

When running a workflow, configuration values are resolved with strict precedence:

1. **Explicit CLI Flags** (highest priority: `--batch-size 256`, `--epochs 50`)
2. **YAML Config File** (values loaded from `--config path.yaml`)
3. **Internal Defaults** (defined in `deconvolve.coretypes.configs.RunConfig`)
