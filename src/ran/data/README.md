# Data

## Config

### `ran.data.config::sigma_to_covariance`

Promote `sigma` (scalar, vector, or matrix) to a (dim, dim) covariance matrix, where `dim` is the dimension of the data.

- scalar -> σ²I
- (dim,) vector -> diag(σ²)
- (dim, dim) matrix -> used as-is

Validates positive-definiteness via Cholesky decomposition.
