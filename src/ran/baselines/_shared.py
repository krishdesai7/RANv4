from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..evaluate import (
    _improvement,
    _js_per_dim,
    _load_splits,
    _triangular_per_dim,
    _wd_per_dim,
)
from ..rantypes import RunConfig, Split, UnfoldingPopulations

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    from ..rantypes import ZXY, DatasetSplits, MetricRecord, Populations


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
    variable_names: tuple[str, ...] = cast(tuple[str, ...], tuple(variables))
    if len(variable_names) != dim:
        raise ValueError(
            f"variables has length {len(variable_names)}, expected dim={dim}"
        )
    return variable_names


def parse_run_config(raw: object) -> RunConfig:
    """Validate a run's config.json into a RunConfig."""
    if not isinstance(raw, dict) or not all(isinstance(k, str) for k in raw):
        raise ValueError("run config must be a JSON object")

    config: dict[str, Any] = cast("dict[str, Any]", raw)
    dim: int = _positive_int(config.get("dim"), "dim")
    n_samples: int = _positive_int(config.get("n_samples"), "n_samples")
    batch_size: int = _positive_int(config.get("batch_size"), "batch_size")
    data_seed: object = config.get("data_seed", 42)
    if type(data_seed) is not int:
        raise ValueError("data_seed must be an integer")

    dataset: Literal["gaussian", "jets"] = _parse_dataset(config)
    variable_names: tuple[str, ...] = _parse_variable_names(config, dataset, dim)

    return RunConfig(
        source=dict(config),
        dataset=dataset,
        dim=dim,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=data_seed,
        variable_names=variable_names,
    )


def _partitioned(data: ZXY, expected_dim: int, label: str) -> Populations:
    """Check the shape assumptions the baselines rely on, then partition.

    `label` names the sample in any error, since a caller partitions several
    and the failures otherwise read identically.
    """
    if data.z.ndim != 2 or data.x.ndim != 2 or data.z.shape != data.x.shape:
        raise ValueError(
            f"{label}: z and x must be identically shaped two-dimensional arrays"
        )
    if data.z.shape[1] != expected_dim:
        raise ValueError(
            f"{label}: array dimension {data.z.shape[1]}, expected dim={expected_dim}"
        )
    if not np.all(np.isfinite(data.z)) or not np.all(np.isfinite(data.x)):
        raise ValueError(f"{label}: z and x values must be finite")
    try:
        return data.partition()
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def prepare_populations(
    splits: DatasetSplits, expected_dim: int
) -> UnfoldingPopulations:
    """Validate the dataset and cut it into the populations baselines need.

    Arrays stay float64; a baseline that needs another dtype (OmniFold wants
    float32 for TensorFlow) casts at its own boundary.
    """
    return UnfoldingPopulations(
        full=_partitioned(splits.select(Split.ALL), expected_dim, "every split"),
        test=_partitioned(splits.select(Split.TEST), expected_dim, "test split"),
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
