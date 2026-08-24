from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ran import cli
from ran.cli import app, baseline_app, sweep_app
from typer.testing import CliRunner

if TYPE_CHECKING:
    from typer.main import Typer

runner = CliRunner()


def _command_names(typer_app: Typer) -> set[str | None]:
    command_names = {command.name for command in typer_app.registered_commands}
    group_names = {group.name for group in typer_app.registered_groups}
    return command_names | group_names


def test_registered_command_trees_are_exact() -> None:
    assert _command_names(app) == {
        "train",
        "evaluate",
        "baseline",
        "sweep",
        "leakage-check",
    }
    assert _command_names(baseline_app) == {"ibu"}
    assert _command_names(sweep_app) == {"ran", "collect"}


@pytest.mark.parametrize(
    "command",
    [
        ("train",),
        ("evaluate",),
        ("baseline", "ibu"),
        ("sweep", "ran"),
        ("sweep", "collect"),
        ("leakage-check",),
    ],
    ids=lambda command: "-".join(command),
)
def test_every_leaf_command_has_help(command) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0


def test_train_converts_typer_values_for_the_workflow(monkeypatch, tmp_path) -> None:
    """`ran.cli` imports `run` at module scope, so patch the name it calls.

    Replacing `sys.modules["ran.workflow"]` would only work if the command
    re-imported on every invocation, which it deliberately no longer does.
    """
    calls = []
    configured_levels = []

    def fake_run(
        batch_size,
        n_samples,
        config,
        dataset,
        variables,
        load_run,
        hidden_units,
        n_layers,
        patience,
        seed,
        data_seed,
    ) -> None:
        calls.append(
            {
                "batch_size": batch_size,
                "n_samples": n_samples,
                "config": config,
                "dataset": dataset,
                "variables": variables,
                "load_run": load_run,
                "hidden_units": hidden_units,
                "n_layers": n_layers,
                "patience": patience,
                "seed": seed,
                "data_seed": data_seed,
            }
        )

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(
        cli, "configure_logging", lambda level: configured_levels.append(level)
    )

    result = runner.invoke(
        app,
        [
            "--log-level",
            "warning",
            "train",
            "--dataset",
            "jets",
            "--var",
            "m",
            "--var",
            "w",
            "--load-run",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["dataset"] is cli.DatasetName.jets
    assert calls[0]["variables"] == frozenset(("m", "w"))
    # Paths stay Paths: `workflow.run` is typed `load_run: Path | None` and
    # opens them directly. Only typer's own wrappers get converted; the
    # DatasetName enum remains an enum, and repeated --var becomes a frozenset.
    assert calls[0]["load_run"] == tmp_path
    assert calls[0]["seed"] == 7
    # LogLevel is a StrEnum of auto() members, so its values are the lowercase
    # names typer shows in --help. configure_logging normalizes the case.
    assert configured_levels == ["warning"]
