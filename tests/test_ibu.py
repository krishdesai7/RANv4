from __future__ import annotations

from typing import Any

import pytest
from ran.baselines import ibu


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset": "gaussian",
        "dim": 2,
        "n_samples": 100,
        "batch_size": 16,
        "data_seed": 7,
    }
    config.update(overrides)
    return config


def test_parse_config_builds_gaussian_variable_names() -> None:
    parsed = ibu._parse_config(_config())

    assert parsed.dataset == "gaussian"
    assert parsed.variable_names == ("dim_0", "dim_1")
    assert parsed.dim == 2
    assert parsed.data_seed == 7


def test_parse_config_requires_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        ibu._parse_config(["not", "a", "mapping"])


def test_parse_config_validates_jet_variable_count() -> None:
    with pytest.raises(ValueError, match=r"variables.*dim"):
        ibu._parse_config(_config(dataset="jets", variables=["mass"], dim=2))


@pytest.mark.parametrize("key", ["dim", "n_samples", "batch_size"])
def test_parse_config_requires_positive_integer_fields(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        ibu._parse_config(_config(**{key: 0}))


def test_parse_config_reports_missing_required_field() -> None:
    config = _config()
    del config["n_samples"]

    with pytest.raises(ValueError, match="n_samples"):
        ibu._parse_config(config)


def test_parse_config_rejects_non_integer_data_seed() -> None:
    with pytest.raises(ValueError, match="data_seed"):
        ibu._parse_config(_config(data_seed="seven"))


def test_parse_config_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        ibu._parse_config(_config(dataset="other"))


def test_parse_config_rejects_non_string_jet_variable() -> None:
    with pytest.raises(ValueError, match=r"variables.*strings"):
        ibu._parse_config(_config(dataset="jets", variables=["mass", 4], dim=2))
