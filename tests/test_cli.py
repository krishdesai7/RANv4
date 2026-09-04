from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ran import cli
from ran.cli import app, baseline_app, sweep_app, uncertainty_app
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
        "uncertainty",
        "leakage-check",
    }
    assert _command_names(baseline_app) == {"ibu"}
    assert _command_names(sweep_app) == {"ran", "collect"}
    assert _command_names(uncertainty_app) == {"run", "collect"}


@pytest.mark.parametrize(
    "command",
    [
        ("train",),
        ("evaluate",),
        ("baseline", "ibu"),
        ("sweep", "ran"),
        ("sweep", "collect"),
        ("uncertainty", "run"),
        ("uncertainty", "collect"),
        ("leakage-check",),
    ],
    ids="-".join,
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
    configured_levels: list[str] = []

    def capture_log_level(*, level: str) -> None:
        configured_levels.append(level)

    def fake_run(
        batch_size,
        n_samples,
        config,
        dataset,
        variables,
        load_run,
        hidden_units,
        n_layers,
        seed,
        data_seed,
        **hyperparameters,
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
                "seed": seed,
                "data_seed": data_seed,
                **hyperparameters,
            }
        )

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "configure_logging", capture_log_level)

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
    # A tuple in canonical order, not a set: these names index columns, and
    # `--var w --var m` must describe the same run as `--var m --var w`.
    assert calls[0]["variables"] == ("m", "w")
    # Paths stay Paths: `workflow.run` is typed `load_run: Path | None` and
    # opens them directly. Only typer's own wrappers get converted; the
    # DatasetName enum remains an enum, and repeated --var becomes a tuple.
    assert calls[0]["load_run"] == tmp_path
    assert calls[0]["seed"] == 7
    # LogLevel is a StrEnum of auto() members, so its values are the lowercase
    # names typer shows in --help. configure_logging normalizes the case.
    assert configured_levels == ["warning"]
    # The training hyperparameters reach `run` too.
    assert calls[0]["n_epochs"] == 100
    assert calls[0]["n_disc_steps"] == 5
    # 3e-5, not 1e-4: measured at +1.22 +- 0.53 points over two paired sweeps
    # (p = 0.022). See "What tuning actually found" in benchmarks/README.md.
    assert calls[0]["lr_g"] == pytest.approx(3e-5)
    assert calls[0]["lr_d"] == pytest.approx(1e-4)
    # 0.015: +4.30 +- 0.88 points on the 12-observable aggregate against 0
    # (p = 0.0017), and admissible on RAN's own selection criterion. See
    # "The dispersion penalty" in benchmarks/README.md.
    assert calls[0]["lambda_dispersion"] == pytest.approx(0.015)
    assert calls[0]["plots"] is True


def test_train_forwards_an_explicit_run_dir(monkeypatch, tmp_path) -> None:
    """A sweep names each run's directory so arms can be told apart.

    Without it every run lands on a second-resolution timestamp and concurrent
    arms overwrite each other -- see TestNewRunDir in tests/test_workflow.py.
    """
    calls = []

    def fake_run(*args, **kwargs) -> None:
        del args
        calls.append(kwargs)

    monkeypatch.setattr(cli, "run", fake_run)

    wanted = tmp_path / "hp_lrg" / "lrg3e-4_seed05"
    result = runner.invoke(app, ["train", "--run-dir", str(wanted)])

    assert result.exit_code == 0
    assert calls[0]["run_dir"] == wanted


def test_train_defaults_to_no_explicit_run_dir(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs) -> None:
        del args
        calls.append(kwargs)

    monkeypatch.setattr(cli, "run", fake_run)

    assert runner.invoke(app, ["train"]).exit_code == 0
    assert calls[0]["run_dir"] is None
