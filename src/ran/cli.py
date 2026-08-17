from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import typer

from .logging_config import configure_logging
from .rantypes import (
    DEFAULT_PURITY_THRESHOLD,
    RUN_DIR,
    SUBSTRUCTURE_VARIABLES,
    DatasetName,
    LogLevel,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Annotated

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
            "-L",
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
    load_run: Annotated[Path | None, typer.Option("--load-run", "-r")] = None,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    patience: Annotated[int, typer.Option("--patience", "-P", min=0)] = 5,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
    from .workflow import run

    run(
        batch_size,
        n_samples,
        config,
        dataset,
        frozenset(variable or SUBSTRUCTURE_VARIABLES),
        load_run,
        hidden_units,
        n_layers,
        patience,
        seed,
        data_seed,
    )


@app.command(name="evaluate")
def evaluate_command(run_dir: Path = RUN_DIR, force: bool = False) -> None:
    from .evaluate import evaluate_runs

    evaluate_runs(run_dir, force)


@baseline_app.command(name="omnifold")
def omnifold_command(
    run_dir: Path = RUN_DIR,
    force: bool = False,
    niter: Annotated[int, typer.Option("--niter", "-i", min=1)] = 3,
    epochs: Annotated[int, typer.Option("--epochs", "-e", min=1)] = 50,
) -> None:
    from .baselines.omnifold import evaluate_runs

    evaluate_runs(run_dir, force, niter, epochs)


@baseline_app.command(name="ibu")
def ibu_command(
    run_dir: Path = RUN_DIR,
    force: bool = False,
    n_iterations: Annotated[int, typer.Option("--niter", "-i", min=1)] = 10,
    purity_threshold: float | np.double = DEFAULT_PURITY_THRESHOLD,
) -> None:
    from .baselines.ibu import evaluate_runs

    purity_threshold = np.double(purity_threshold)
    evaluate_runs(
        run_dir,
        force,
        n_iterations,
        purity_threshold,
    )


@sweep_app.command(name="ran")
def sweep_ran_command(
    s_index: Annotated[int, typer.Option("--s-index", "-s", min=0)],
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir", "-d")],
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    n_points: Annotated[int, typer.Option("--n-points", "-p", min=1)] = 25,
    seed: int = 42,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    ran_epochs: Annotated[int, typer.Option("--ran-epochs", "-e", min=1)] = 100,
    init_seed: Annotated[int | None, typer.Option("--init-seed", "-I")] = None,
) -> None:
    from .experiments.cubic_sweep import run_ran

    run_ran(
        s_index,
        sweep_dir,
        n_samples,
        n_points,
        seed,
        batch_size,
        ran_epochs,
        init_seed,
    )


@sweep_app.command(name="omnifold")
def sweep_omnifold_command(
    s_index: Annotated[int, typer.Option("--s-index", "-s", min=0)],
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir", "-d")],
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    n_points: Annotated[int, typer.Option("--n-points", "-p", min=1)] = 25,
    seed: int = 42,
    omnifold_niter: Annotated[int, typer.Option("--niter", "-i", min=1)] = 3,
    omnifold_epochs: Annotated[int, typer.Option("--epochs", "-e", min=1)] = 50,
    omnifold_batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", min=1)
    ] = 512,
) -> None:
    from .experiments.cubic_sweep import run_omnifold

    run_omnifold(
        s_index,
        sweep_dir,
        n_samples,
        n_points,
        seed,
        omnifold_niter,
        omnifold_epochs,
        omnifold_batch_size,
    )


@sweep_app.command(name="collect")
def sweep_collect_command(
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir", "-d")],
    n_points: Annotated[int, typer.Option("--n-points", "-p", min=1)] = 25,
) -> None:
    from .experiments.cubic_sweep import collect

    collect(sweep_dir, n_points)


@app.command(name="leakage-check")
def leakage_check_command(
    poison: Annotated[bool, typer.Option("--poison/--clean", "-X/")] = False,
    seed: int = 42,
    init_seed: int = 0,
) -> None:
    from .leakage import run_leakage_check

    run_leakage_check(poison, seed, init_seed)
