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
from ..rantypes import DatasetName, RunConfig, Split, UnfoldingPopulations

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from ..rantypes import ZXY, DatasetSplits, MetricRecord, Populations


def _positive_int(value: object, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _parse_dataset(raw: dict[str, Any]) -> DatasetName:
    value: object = raw.get("dataset", DatasetName.gaussian.value)
    try:
        return DatasetName(value=value)
    except ValueError as error:
        known: str = ", ".join(name.value for name in DatasetName)
        raise ValueError(
            f"Unknown dataset {value!r}, expected one of {known}"
        ) from error


def _parse_variable_names(
    raw: dict[str, Any], dataset: DatasetName, dim: int, /
) -> tuple[str, ...]:
    if dataset == DatasetName.gaussian:
        return tuple(f"dim_{i}" for i in range(dim))

    variables: object = raw.get("variables")
    if not isinstance(variables, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in variables
    ):
        raise ValueError("variables must be a sequence of nonempty strings")
    variable_names: tuple[str, ...] = cast(typ=tuple[str, ...], val=tuple(variables))
    if len(variable_names) != dim:
        raise ValueError(
            f"variables has length {len(variable_names)}, expected dim={dim}"
        )
    return variable_names


def parse_run_config(raw: object) -> RunConfig:
    if not isinstance(raw, dict) or not all(isinstance(k, str) for k in raw):
        raise ValueError("run config must be a JSON object")

    config: dict[str, Any] = cast(typ="dict[str, Any]", val=raw)
    dim: int = _positive_int(value=config.get("dim"), key="dim")
    n_samples: int = _positive_int(value=config.get("n_samples"), key="n_samples")
    batch_size: int = _positive_int(value=config.get("batch_size"), key="batch_size")
    data_seed: object = config.get("data_seed", 42)
    if not isinstance(data_seed, int):
        raise TypeError("data_seed must be an integer")

    dataset: DatasetName = _parse_dataset(raw=config)
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


def _partitioned[T: np.floating](
    data: ZXY[T], expected_dim: int, label: str, /
) -> Populations[T]:
    if data.z.ndim != 2 or data.x.ndim != 2 or data.z.shape != data.x.shape:
        raise ValueError(
            f"{label}: z and x must be identically shaped two-dimensional arrays"
        )
    if data.z.shape[1] != expected_dim:
        raise ValueError(
            f"{label}: array dimension {data.z.shape[1]}, expected dim={expected_dim}"
        )
    if not np.all(a=np.isfinite(data.z)) or not np.all(a=np.isfinite(data.x)):
        raise ValueError(f"{label}: z and x values must be finite")
    try:
        return data.partition()
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def prepare_populations[T: np.floating](
    splits: DatasetSplits[T], expected_dim: int
) -> UnfoldingPopulations[T]:
    return UnfoldingPopulations(
        full=_partitioned(splits.select(Split.ALL), expected_dim, "every split"),
        test=_partitioned(splits.select(Split.TEST), expected_dim, "test split"),
    )


def load_populations(config: RunConfig) -> UnfoldingPopulations[np.double]:
    """The run's dataset as populations, at the float64 RAN generated it in.

    Baselines that need another precision call `astype` at their own boundary.
    """
    return prepare_populations(
        _load_splits(config=config.source), expected_dim=config.dim
    )


def evaluate_dimension[T: np.floating](
    reference: NDArray[T],
    comparison: NDArray[T],
    weights: NDArray[T],
) -> MetricRecord:
    wasserstein_before: float = _wd_per_dim(ref=reference, comp=comparison)[0]
    wasserstein_after: float = _wd_per_dim(
        ref=reference, comp=comparison, weights=weights
    )[0]
    jensenshannon_before: float = _js_per_dim(ref=reference, comp=comparison)[0]
    jensenshannon_after: float = _js_per_dim(
        ref=reference, comp=comparison, weights=weights
    )[0]
    triangular_before: float = _triangular_per_dim(ref=reference, comp=comparison)[0]
    triangular_after: float = _triangular_per_dim(
        ref=reference, comp=comparison, weights=weights
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
