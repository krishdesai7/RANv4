import subprocess  # ruff: ignore[suspicious-subprocess-import] -- import isolation
import sys
from types import ModuleType

import pytest
from ran import cli
from ran.cli import app, baseline_app, sweep_app
from typer.main import Typer
from typer.testing import CliRunner

runner = CliRunner()


def _command_names(typer_app: Typer):
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
    assert _command_names(baseline_app) == {"omnifold", "ibu"}
    assert _command_names(sweep_app) == {"ran", "omnifold", "collect"}


@pytest.mark.parametrize(
    "command",
    [
        ("train",),
        ("evaluate",),
        ("baseline", "omnifold"),
        ("baseline", "ibu"),
        ("sweep", "ran"),
        ("sweep", "omnifold"),
        ("sweep", "collect"),
        ("leakage-check",),
    ],
    ids=lambda command: "-".join(command),
)
def test_every_leaf_command_has_help(command) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0


def test_importing_cli_does_not_commit_a_keras_backend() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ran.cli; assert 'keras' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_train_converts_typer_values_for_the_workflow(monkeypatch, tmp_path) -> None:
    calls = []
    configured_levels = []
    fake_workflow = ModuleType("ran.workflow")
    fake_workflow.__dict__["run"] = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "ran.workflow", fake_workflow)
    monkeypatch.setattr(cli, "configure_logging", configured_levels.append)

    result = runner.invoke(
        app,
        [
            "--log-level",
            "warning",
            "train",
            "--dataset",
            "jets",
            "--variable",
            "m",
            "--variable",
            "w",
            "--load-run",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["dataset"] == "jets"
    assert calls[0]["variables"] == ("m", "w")
    assert calls[0]["load_run"] == str(tmp_path)
    assert calls[0]["seed"] == 7
    assert configured_levels == ["WARNING"]
