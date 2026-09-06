from __future__ import annotations

from pathlib import Path  # ruff: ignore[typing-only-standard-library-import]
from typing import Annotated

import numpy as np
import typer

from .baselines import evaluate_runs as ibu_evaluate_runs
from .evaluate import evaluate_runs
from .experiments import run_ran
from .leakage import run_leakage_check
from .logging_config import configure_logging
from .rantypes import (
    DEFAULT_PURITY_THRESHOLD,
    POISON_SENTINEL,
    RUN_DIR,
    SUBSTRUCTURE_VARIABLES,
    DatasetName,
    LogLevel,
)
from .workflow import run

baseline_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
sweep_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
uncertainty_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
app.add_typer(
    typer_instance=baseline_app, name="baseline", help="Run comparison baselines."
)
app.add_typer(
    typer_instance=sweep_app, name="sweep", help="Run cubic-response sweep steps."
)
app.add_typer(
    typer_instance=uncertainty_app,
    name="uncertainty",
    help="Bootstrap x seed variance decomposition.",
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


def _canonical_variables(chosen: list[str] | None, /) -> tuple[str, ...]:
    """Fix the jet column order once, here, so nothing downstream has to guess.

    Repeated `--var` arrives as a list in whatever order it was typed. Sorting
    it into `SUBSTRUCTURE_VARIABLES` order means `--var w --var m` and
    `--var m --var w` describe the same run --- same columns, same cache key,
    same `config.json` --- rather than two runs that differ only in a permutation
    nobody chose.
    """
    if not chosen:
        return SUBSTRUCTURE_VARIABLES
    wanted: set[str] = set(chosen)
    return tuple(v for v in SUBSTRUCTURE_VARIABLES if v in wanted) + tuple(
        # Unknown names pass through in order so `load_jet_dataset` is the one
        # place that reports them, with the list of what it does accept.
        v
        for v in dict.fromkeys(chosen)
        if v not in SUBSTRUCTURE_VARIABLES
    )


@app.command(name="train")
def train_command(
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    config: Path | None = None,
    dataset: Annotated[
        DatasetName, typer.Option("--dataset", "-D")
    ] = DatasetName.gaussian,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    load_run: Annotated[Path | None, typer.Option("--load-run", "-r")] = None,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 100,
    n_disc_steps: Annotated[int, typer.Option("--n-disc-steps", "-k", min=1)] = 5,
    lr_g: Annotated[float, typer.Option("--lr-g", min=0.0)] = 3e-5,
    lr_d: Annotated[float, typer.Option("--lr-d", min=0.0)] = 1e-4,
    lambda_dispersion: Annotated[
        float,
        typer.Option(
            "--lambda-dispersion",
            min=0.0,
            help="Penalty on the variance of g's weights. 0 disables it.",
        ),
    ] = 0.015,
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Draw figures. Off is for sweeps: metrics still run.",
        ),
    ] = True,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            help="Where to save this run. Default is a timestamp under runs/.",
        ),
    ] = None,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
    run(
        batch_size,
        n_samples,
        config,
        dataset,
        _canonical_variables(variable),
        load_run,
        hidden_units,
        n_layers,
        seed,
        data_seed,
        n_epochs=n_epochs,
        n_disc_steps=n_disc_steps,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_dispersion=lambda_dispersion,
        plots=plots,
        run_dir=run_dir,
    )


@app.command(name="evaluate")
def evaluate_command(run_dir: Path = RUN_DIR, force: bool = False) -> None:
    evaluate_runs(run_dir, force)


@baseline_app.command(name="ibu")
def ibu_command(
    run_dir: Path = RUN_DIR,
    force: bool = False,
    n_iterations: Annotated[int, typer.Option("--niter", "-i", min=1)] = 10,
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
) -> None:
    ibu_evaluate_runs(
        run_dir,
        force,
        n_iterations,
        purity_threshold=np.double(purity_threshold),
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
    _ = run_ran(
        s_index,
        sweep_dir,
        n_samples,
        n_points,
        seed,
        batch_size,
        ran_epochs,
        init_seed,
    )


@sweep_app.command(name="collect")
def sweep_collect_command(
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir", "-d")],
    n_points: Annotated[int, typer.Option("--n-points", "-p", min=1)] = 25,
) -> None:
    from .experiments.cubic_sweep import collect

    collect(sweep_dir, n_points)


@uncertainty_app.command(name="run")
def uncertainty_run_command(
    cell: Annotated[int, typer.Option("--cell", "-c", min=0)],
    design_dir: Annotated[Path, typer.Option("--design-dir", "-d")],
    n_datasets: Annotated[int, typer.Option("--n-datasets", "-B", min=2)] = 8,
    n_seeds: Annotated[int, typer.Option("--n-seeds", "-S", min=2)] = 8,
    n_eval: Annotated[int, typer.Option("--n-eval", min=1)] = 100_000,
    dataset: Annotated[DatasetName, typer.Option("--dataset", "-D")] = DatasetName.jets,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    config: Path | None = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 100,
    n_disc_steps: Annotated[int, typer.Option("--n-disc-steps", "-k", min=1)] = 5,
    lr_g: Annotated[float, typer.Option("--lr-g", min=0.0)] = 3e-5,
    lr_d: Annotated[float, typer.Option("--lr-d", min=0.0)] = 1e-4,
    lambda_dispersion: Annotated[
        float, typer.Option("--lambda-dispersion", min=0.0)
    ] = 0.015,
    data_seed: int = 42,
    init_seed: int = 0,
) -> None:
    """Train one (bootstrap dataset, init seed) cell of the design."""
    from .uncertainty import DesignSpec, run_cell

    _ = run_cell(
        cell,
        design_dir,
        DesignSpec(n_datasets, n_seeds, data_seed, init_seed),
        dataset=dataset,
        variables=_canonical_variables(variable),
        config=config,
        n_samples=n_samples,
        n_eval=n_eval,
        batch_size=batch_size,
        hidden_units=hidden_units,
        n_layers=n_layers,
        n_epochs=n_epochs,
        n_disc_steps=n_disc_steps,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_dispersion=lambda_dispersion,
    )


@uncertainty_app.command(name="collect")
def uncertainty_collect_command(
    design_dir: Annotated[Path, typer.Option("--design-dir", "-d")],
    n_datasets: Annotated[int, typer.Option("--n-datasets", "-B", min=2)] = 8,
    n_seeds: Annotated[int, typer.Option("--n-seeds", "-S", min=2)] = 8,
    n_bins: Annotated[int, typer.Option("--n-bins", min=2)] = 20,
    data_seed: int = 42,
    init_seed: int = 0,
) -> None:
    """Decompose a finished design and write its table, npz and figure."""
    from .uncertainty import DesignSpec, collect

    _ = collect(
        design_dir,
        DesignSpec(n_datasets, n_seeds, data_seed, init_seed),
        n_bins=n_bins,
    )


@app.command(name="leakage-check")
def leakage_check_command(
    poison: Annotated[bool, typer.Option("--poison/--clean", "-X/")] = False,
    sentinel: Annotated[float, typer.Option("--sentinel", "-S")] = POISON_SENTINEL,
    seed: int = 42,
    init_seed: int = 0,
) -> None:
    run_leakage_check(poison, sentinel, seed, init_seed)
