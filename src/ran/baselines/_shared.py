"""Data handling shared by the IBU and OmniFold baselines.

Both baselines answer the same question -- reweight MC to match observed data,
then score the result -- so they need the same run config, the same event
populations, and the same metric record. Only the unfolding method differs.

This module must stay free of `keras` at import time. `ran.baselines.omnifold`
pins ``KERAS_BACKEND=tensorflow`` before importing keras, and it imports from
here; pulling keras in transitively would fix the backend to jax first and break
OmniFold. `ran.evaluate` is safe to import for the same reason -- it defers its
own keras import into the two functions that need it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..evaluate import (
    _improvement,
    _js_per_dim,
    _load_splits,
    _triangular_per_dim,
    _wd_per_dim,
)
from ..rantypes import RunConfig, UnfoldingPopulations

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    from ..data import ArrayDataset
    from ..rantypes import DatasetSplits, MetricRecord


def _positive_int(value: object, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _parse_dataset(raw: dict[str, Any]) -> Literal["gaussian", "jets"]:
    dataset: object = raw.get("dataset", "gaussian")
    if dataset == "gaussian":
        return "gaussian"
    if dataset == "jets":
        return "jets"
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _parse_variable_names(
    raw: dict[str, Any], dataset: Literal["gaussian", "jets"], dim: int
) -> tuple[str, ...]:
    if dataset == "gaussian":
        return tuple(f"dim_{i}" for i in range(dim))

    variables: object = raw.get("variables")
    if not isinstance(variables, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in variables
    ):
        raise ValueError("variables must be a sequence of nonempty strings")
    variable_names = tuple(variables)
    if len(variable_names) != dim:
        raise ValueError(
            f"variables has length {len(variable_names)}, expected dim={dim}"
        )
    return variable_names


def parse_run_config(raw: object) -> RunConfig:
    """Validate a run's config.json into a RunConfig."""
    if not isinstance(raw, dict):
        # ValueError, not TypeError, to match every other validation failure in
        # this module -- callers catch one exception type for a bad config.
        raise ValueError("run config must be a JSON object")  # ruff: ignore[type-check-without-type-error]

    dim: int = _positive_int(raw.get("dim"), "dim")
    n_samples: int = _positive_int(raw.get("n_samples"), "n_samples")
    batch_size: int = _positive_int(raw.get("batch_size"), "batch_size")
    data_seed: object = raw.get("data_seed", 42)
    if type(data_seed) is not int:
        raise ValueError("data_seed must be an integer")

    dataset = _parse_dataset(raw)
    variable_names = _parse_variable_names(raw, dataset, dim)

    return RunConfig(
        source=dict(raw),
        dataset=dataset,
        dim=dim,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=data_seed,
        variable_names=variable_names,
    )


def _validated_arrays(
    split: ArrayDataset, expected_dim: int
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.ubyte]]:
    z, x, y = split.as_arrays()
    if z.ndim != 2 or x.ndim != 2 or z.shape != x.shape:
        raise ValueError("z and x must be identically shaped two-dimensional arrays")
    if z.shape[1] != expected_dim:
        raise ValueError(f"array dimension {z.shape[1]}, expected dim={expected_dim}")
    if y.ndim != 1 or y.shape[0] != z.shape[0]:
        raise ValueError("y must be one-dimensional with one label per row")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(x)):
        raise ValueError("z and x values must be finite")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be only zero or one")
    return z, x, y


def prepare_populations(
    splits: DatasetSplits, expected_dim: int
) -> UnfoldingPopulations:
    """Validate dataset arrays and separate response and evaluation populations.

    Arrays stay float64; a baseline that needs another dtype (OmniFold wants
    float32 for TensorFlow) casts at its own boundary.
    """
    arrays = [
        _validated_arrays(split, expected_dim)
        for split in (splits.train, splits.val, splits.test)
    ]

    z_all: NDArray[np.double] = np.concatenate([item[0] for item in arrays], axis=0)
    x_all: NDArray[np.double] = np.concatenate([item[1] for item in arrays], axis=0)
    y_all: NDArray[np.ubyte] = np.concatenate([item[2] for item in arrays], axis=0)
    z_test, x_test, y_test = arrays[-1]

    response_mask = y_all == 0
    observed_mask = y_all == 1
    test_mc_mask = y_test == 0
    test_data_mask = y_test == 1
    if not np.any(response_mask):
        raise ValueError("response MC population must not be empty")
    if not np.any(observed_mask):
        raise ValueError("observed data population must not be empty")
    if not np.any(test_mc_mask):
        raise ValueError("test MC population must not be empty")
    if not np.any(test_data_mask):
        raise ValueError("test data population must not be empty")

    return UnfoldingPopulations(
        response_gen=z_all[response_mask],
        response_sim=x_all[response_mask],
        observed_reco=x_all[observed_mask],
        test_data_gen=z_test[test_data_mask],
        test_data_reco=x_test[test_data_mask],
        test_mc_gen=z_test[test_mc_mask],
        test_mc_reco=x_test[test_mc_mask],
    )


def load_populations(config: RunConfig) -> UnfoldingPopulations:
    """Rebuild the run's dataset and split it into the baseline populations."""
    return prepare_populations(_load_splits(config.source), config.dim)


def evaluate_dimension(
    reference: NDArray[np.double],
    comparison: NDArray[np.double],
    weights: NDArray[np.floating],
) -> MetricRecord:
    """Score one dimension before and after reweighting `comparison`.

    `reference` and `comparison` are 1D slices, so each per-dim metric returns a
    single-element list. `weights` is any float precision -- OmniFold's come
    back from TensorFlow as float32 and are stored that way.
    """
    wasserstein_before: float = _wd_per_dim(reference, comparison)[0]
    wasserstein_after: float = _wd_per_dim(reference, comparison, weights=weights)[0]
    jensenshannon_before: float = _js_per_dim(reference, comparison)[0]
    jensenshannon_after: float = _js_per_dim(reference, comparison, weights=weights)[0]
    triangular_before: float = _triangular_per_dim(reference, comparison)[0]
    triangular_after: float = _triangular_per_dim(
        reference, comparison, weights=weights
    )[0]
    return {
        "wasserstein_before": wasserstein_before,
        "wasserstein_after": wasserstein_after,
        "wasserstein_improvement_pct": _improvement(
            wasserstein_before, wasserstein_after
        ),
        "jensenshannon_before": jensenshannon_before,
        "jensenshannon_after": jensenshannon_after,
        "jensenshannon_improvement_pct": _improvement(
            jensenshannon_before, jensenshannon_after
        ),
        "triangular_before": triangular_before,
        "triangular_after": triangular_after,
        "triangular_improvement_pct": _improvement(triangular_before, triangular_after),
    }
