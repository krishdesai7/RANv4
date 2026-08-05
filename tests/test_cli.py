import subprocess
import sys
from types import ModuleType

from typer.testing import CliRunner

import ran.cli as cli
from ran.cli import app

runner = CliRunner()


def test_root_help_lists_unified_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "evaluate", "baseline", "sweep", "leakage-check"):
        assert command in result.stdout


def test_baseline_help_lists_both_methods():
    result = runner.invoke(app, ["baseline", "--help"])
    assert result.exit_code == 0
    assert "omnifold" in result.stdout
    assert "ibu" in result.stdout


def test_sweep_help_lists_all_actions():
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    for command in ("ran", "omnifold", "collect"):
        assert command in result.stdout


def test_importing_cli_does_not_commit_a_keras_backend():
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


def test_train_converts_typer_values_for_the_workflow(monkeypatch, tmp_path):
    calls = []
    configured_levels = []
    fake_workflow = ModuleType("ran.workflow")
    fake_workflow.run = lambda **kwargs: calls.append(kwargs)
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
