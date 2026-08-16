from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .logging_config import configure_logging
from .rantypes import SUBSTRUCTURE_VARIABLES, DatasetName, LogLevel

baseline_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
sweep_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
app.add_typer(
    typer_instance=baseline_app, name="baseline", help="Run comparison baselines."
)
app.add_typer(
    typer_instance=sweep_app, name="sweep", help="Run cubic-response sweep steps."
)


@app.callback()
def configure(
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-l",
            case_sensitive=False,
            envvar="RAN_LOG_LEVEL",
            help="Application log level.",
        ),
    ] = LogLevel.info,
) -> None:
    configure_logging(level=log_level.value)


@app.command(name="train")
def train_command(
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    config: Path | None = None,
    dataset: DatasetName = DatasetName.gaussian,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    load_run: Annotated[Path | None, typer.Option()] = None,
    hidden_units: Annotated[int, typer.Option(min=1)] = 64,
    n_layers: Annotated[int, typer.Option(min=1)] = 2,
    patience: Annotated[int, typer.Option(min=1)] = 5,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
    from ran.workflow import run

    run(
        batch_size=batch_size,
        n_samples=n_samples,
        config=config,
        dataset=dataset.value,
        variables=frozenset(variable or SUBSTRUCTURE_VARIABLES),
        load_run=load_run,
        hidden_units=hidden_units,
        n_layers=n_layers,
        patience=patience,
        seed=seed,
        data_seed=data_seed,
    )


@app.command("evaluate")
def evaluate_command(run_dir: Path = Path("runs"), force: bool = False) -> None:
    from ran.evaluate import evaluate_runs

    evaluate_runs(run_dir=run_dir, force=force)


@baseline_app.command("omnifold")
def omnifold_command(
    run_dir: Path = Path("runs"),
    force: bool = False,
    niter: Annotated[int, typer.Option(min=1)] = 3,
    epochs: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    from ran.baselines.omnifold import evaluate_runs

    evaluate_runs(run_dir=run_dir, force=force, niter=niter, epochs=epochs)


@baseline_app.command("ibu")
def ibu_command(
    run_dir: Path = Path("runs"),
    force: bool = False,
    n_iterations: Annotated[int, typer.Option(min=1)] = 10,
    purity_threshold: float = 0.7071067811865476,
) -> None:
    from numpy import double

    from ran.baselines.ibu import evaluate_runs

    evaluate_runs(
        run_dir=run_dir,
        force=force,
        n_iterations=n_iterations,
        purity_threshold=double(purity_threshold),
    )


@sweep_app.command("ran")
def sweep_ran_command(
    s_index: Annotated[int, typer.Option()],
    sweep_dir: Annotated[Path, typer.Option()],
    n_samples: Annotated[int, typer.Option(min=1)] = 500_000,
    n_points: Annotated[int, typer.Option(min=1)] = 25,
    seed: int = 42,
    batch_size: Annotated[int, typer.Option(min=1)] = 1024,
    ran_epochs: Annotated[int, typer.Option(min=1)] = 100,
    init_seed: int | None = None,
) -> None:
    from ran.experiments.cubic_sweep import run_ran

    run_ran(
        s_index=s_index,
        sweep_dir=sweep_dir,
        n_samples=n_samples,
        n_points=n_points,
        seed=seed,
        batch_size=batch_size,
        ran_epochs=ran_epochs,
        init_seed=init_seed,
    )


@sweep_app.command("omnifold")
def sweep_omnifold_command(
    s_index: Annotated[int, typer.Option()],
    sweep_dir: Annotated[Path, typer.Option()],
    n_samples: Annotated[int, typer.Option(min=1)] = 500_000,
    n_points: Annotated[int, typer.Option(min=1)] = 25,
    seed: int = 42,
    omnifold_niter: Annotated[int, typer.Option(min=1)] = 3,
    omnifold_epochs: Annotated[int, typer.Option(min=1)] = 50,
    omnifold_batch_size: Annotated[int, typer.Option(min=1)] = 512,
) -> None:
    from ran.experiments.cubic_sweep import run_omnifold

    run_omnifold(
        s_index=s_index,
        sweep_dir=sweep_dir,
        n_samples=n_samples,
        n_points=n_points,
        seed=seed,
        omnifold_niter=omnifold_niter,
        omnifold_epochs=omnifold_epochs,
        omnifold_batch_size=omnifold_batch_size,
    )


@sweep_app.command("collect")
def sweep_collect_command(
    sweep_dir: Annotated[Path, typer.Option()],
    n_points: Annotated[int, typer.Option(min=1)] = 25,
) -> None:
    from ran.experiments.cubic_sweep import collect

    collect(sweep_dir=sweep_dir, n_points=n_points)


@app.command("leakage-check")
def leakage_check_command(
    poison: Annotated[bool, typer.Option("--poison/--clean")] = False,
    seed: int = 42,
    init_seed: int = 0,
) -> None:
    from ran.leakage import run_leakage_check

    run_leakage_check(poison=poison, seed=seed, init_seed=init_seed)
