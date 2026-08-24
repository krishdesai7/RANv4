"""Tests for the --load-run reload path.

This path had two independent breaks that the suite never saw, both found by a
cluster smoke run rather than by CI:

  - `run()` handed `parse_run_config` the raw text of config.json, but that
    function validates an already-decoded object, so every reload raised
    "run config must be a JSON object";
  - the Gaussian params were then read with the wrong key spelling.

Nothing here trains. The reload path does not call `train` at all, and the
model load, the plots and the metrics are stubbed -- what is exercised is the
config parsing and the dataset rebuild, which is exactly where the breaks were.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ran import workflow
from ran.data import parse_gaussian_config
from ran.rantypes import DatasetName

if TYPE_CHECKING:
    from pathlib import Path

    from ran.rantypes import DatasetSplits

CONFIG_2D = """
mu_gen: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_gen:
  - [1.0, -0.9]
  - [-0.9, 2.25]
sigma_true:
  - [0.81, -0.702]
  - [-0.702, 1.69]
sigma_detector: [0.5, 0.8]
"""


@dataclass
class Recorded:
    """What the stubbed-out tail of `run()` was asked to do."""

    plots: list[Path | None] = field(default_factory=list)
    evaluated: tuple[Path, bool] | None = None


@pytest.fixture
def stub_heavy(monkeypatch) -> Recorded:
    """Stub the parts a reload does not need to prove config handling works."""
    recorded = Recorded()

    def fake_load_artifacts(run_dir: Path):
        del run_dir
        return lambda z: np.ones((len(z), 1)), {"train_d": [0.7]}

    def fake_plot(*args, save_path: Path | None = None, **kwargs) -> None:
        del args, kwargs
        recorded.plots.append(save_path)

    def fake_evaluate(run_dir: Path, force: bool) -> None:
        recorded.evaluated = (run_dir, force)

    monkeypatch.setattr(workflow, "_load_artifacts", fake_load_artifacts)
    for name in ("plot_detector_level", "plot_particle_level", "plot_losses"):
        monkeypatch.setattr(workflow, name, fake_plot)
    monkeypatch.setattr(workflow, "evaluate_run", fake_evaluate)
    return recorded


def _reload(run_dir: Path) -> None:
    """Drive `run()` down the --load-run path.

    `run()` keeps no defaults of its own -- the CLI owns them -- so every
    argument has to be named. The values below are placeholders a reload is
    meant to replace with what config.json recorded, deliberately unlike the
    ones `_write_run` writes so that
    `test_load_run_forwards_recorded_seed_and_size` fails if it stops doing so.
    """
    workflow.run(
        batch_size=8,
        n_samples=32,
        config=None,
        dataset=DatasetName.gaussian,
        variables=(),
        load_run=run_dir,
        hidden_units=4,
        n_layers=1,
        patience=1,
        seed=None,
        data_seed=0,
    )


def _write_run(tmp_path, gaussian_params: dict, **overrides):
    run_dir = tmp_path / "runs" / "2026-08-06T000000Z"
    run_dir.mkdir(parents=True)
    config = {
        "batch_size": 64,
        "n_samples": 400,
        "dim": 2,
        "dataset": "gaussian",
        "seed": 0,
        "data_seed": 42,
        "gaussian_params": gaussian_params,
        **overrides,
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def test_load_run_reads_a_config_written_by_save_run(
    tmp_path, monkeypatch, stub_heavy
) -> None:
    """The round trip that matters: what _save_run writes, run() must read."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg.yaml").write_text(CONFIG_2D)
    params = parse_gaussian_config(tmp_path / "cfg.yaml")
    run_dir = _write_run(tmp_path, params.model_dump())

    _reload(run_dir)

    assert len(stub_heavy.plots) == 3
    assert stub_heavy.evaluated == (run_dir, False)


def test_load_run_reads_master_era_sigma_keys(
    tmp_path, monkeypatch, stub_heavy
) -> None:
    """Runs written before the type refactor stored covariances as sigma_*."""
    monkeypatch.chdir(tmp_path)
    run_dir = _write_run(
        tmp_path,
        {
            "dim": 2,
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [[1.0, -0.9], [-0.9, 2.25]],
            "sigma_true": [[0.81, -0.702], [-0.702, 1.69]],
            "sigma_detector": [[0.25, 0.0], [0.0, 0.64]],
        },
    )

    _reload(run_dir)

    assert len(stub_heavy.plots) == 3


@pytest.mark.usefixtures("stub_heavy")
def test_load_run_forwards_recorded_seed_and_size(tmp_path, monkeypatch) -> None:
    """A reload must rebuild the split the run trained on, not a fresh one.

    `data_seed` and `n_samples` come from config.json, so the events the plots
    and metrics see are the ones the generator was fitted against.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg.yaml").write_text(CONFIG_2D)
    params = parse_gaussian_config(tmp_path / "cfg.yaml")
    run_dir = _write_run(tmp_path, params.model_dump(), data_seed=7, n_samples=600)

    seen: dict[str, object] = {}
    real = workflow.RANDataset

    class Recording(real):
        def __init__(self, *args, **kwargs) -> None:
            seen["seed"] = kwargs.get("seed")
            super().__init__(*args, **kwargs)

        def generate_gaussian_dataset(self, *args, **kwargs) -> DatasetSplits:
            seen["n_samples"] = kwargs.get("n_samples")
            return super().generate_gaussian_dataset(*args, **kwargs)

    monkeypatch.setattr(workflow, "RANDataset", Recording)
    _reload(run_dir)

    assert seen == {"seed": 7, "n_samples": 600}
