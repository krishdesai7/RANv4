# Data

## Config

Configuration files for Gaussian datasets are YAML files with the following structure:

```yaml
mu_gen: float |list[float]
mu_true: float | list[float]
sigma_gen: float | list[float] | list[list[float]]
sigma_true: float | list[float] | list[list[float]]
sigma_detector: float | list[float] | list[list[float]]
```

### `ran.data.config::sigma_to_covariance`

Promote `sigma` (scalar, vector, or matrix) to a (dim, dim) covariance matrix, where `dim` is the dimension of the data.

- scalar -> σ²I
- (dim,) vector -> diag(σ²)
- (dim, dim) matrix -> used as-is

Validates positive-definiteness via Cholesky decomposition.

### `ran.data.config::parse_gaussian_config`

Parse a Gaussian YAML config file and return a dictionary with the following keys:

- `dim: int` - The dimension of the data.
- `mu_gen: NDArray[np.double]` - The mean of the generated data.
- `mu_true: NDArray[np.double]` - The mean of the true data.
- `cov_gen: NDArray[np.double]` - The covariance matrix of the generated data.
- `cov_true: NDArray[np.double]` - The covariance matrix of the true data.
- `cov_detector: NDArray[np.double]` - The covariance matrix of the detector.
