"""The OmniFold baseline, run in a quarantined subprocess.

OmniFold needs TensorFlow; TensorFlow publishes no wheels for this project's
Python floor; and Keras binds its backend once per interpreter. All three are
intra-interpreter constraints, so all three dissolve at a process boundary:
`_omnifold_worker.py` carries a PEP 723 header, `uv run --no-project`
provisions Python 3.13 and TensorFlow for it, and the two halves exchange one
`.npz` file. Nothing in this module imports TensorFlow, and nothing in the
worker can import `ran`.

This half does what every other baseline does --- read a run's `config.json`,
rebuild its populations, score a weight vector against the same metrics
`ran evaluate` uses --- and the symmetry is the point. The comparison is only
worth anything if both arms are scored by the same code, which is why this lives
here rather than in a separate repository with its own vendored copy of
`ran.evaluate` drifting away from this one.

Two things differ from `ibu.py`, both forced by the subprocess:

**`uv` must be on `PATH` at runtime.** It is what provisions the worker, so a
checkout without it cannot run this baseline. The failure is turned into a
readable message rather than a `FileNotFoundError` from deep inside
`subprocess`, because the fix is an install and not a bug.

**A silent CPU fallback is the expected failure, not an exception.** A
TensorFlow that cannot load its CUDA libraries reports no GPU and runs anyway,
returning correct weights tens of times slower --- and on Perlmutter that is the
default state of the environment without `module load cudatoolkit/12.9`. The
worker therefore reports the device it used and this module warns when it was
the CPU. See `benchmarks/gpu_coexistence.py`, which measures it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import timing
from ..evaluate import apply_to_runs, render_metrics
from ..rantypes import artifacts_dir
from ._shared import evaluate_dimension, load_populations, parse_run_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager
    from logging import Logger
    from typing import Any

    from numpy.typing import NDArray

    from ..rantypes import (
        EventArray,
        MetricRecord,
        RunConfig,
        UnfoldingPopulations,
    )

logger: Logger = logging.getLogger(name=__name__)

DEFAULT_N_ITERATIONS: int = 3
DEFAULT_N_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 512

# Generous because it bounds a full training run on a possibly-cold environment,
# not a single call: uv may be resolving and downloading ~3.5GB of CUDA wheels on
# the first invocation. A bound still exists so a hung worker fails the job
# rather than holding the allocation to its wall clock.
WORKER_TIMEOUT_SECONDS: float = 10_800.0


def worker_script() -> AbstractContextManager[Path]:
    """A real filesystem path to the worker, for the lifetime of the context.

    `uv run` needs a path on disk, which a `Traversable` is not obliged to be ---
    hence `as_file` rather than reaching into `__file__`. For an ordinary wheel
    install it is already a real path and this is free; from a zipimport it
    extracts, which is why the caller must treat it as a context manager and not
    stash the path.
    """
    return resources.as_file(
        resources.files(anchor="ran") / "baselines" / "_omnifold_worker.py"
    )


def _worker_env() -> dict[str, str]:
    """The worker's environment, with its own directory off `sys.path`.

    `PYTHONSAFEPATH` is load-bearing and the reason is a name collision this
    module creates. A script's own directory goes on `sys.path[0]`, and the
    worker's own directory is this one --- which contains `omnifold.py`. So the
    worker's `from omnifold import MLP, DataLoader, MultiFold` resolved to *this
    module* rather than to the installed package, and then died on
    `from .. import timing` with "attempted relative import with no known parent
    package": a confusing error a long way from its cause.

    `PYTHONSAFEPATH=1` (3.11+) stops the interpreter prepending the script
    directory, which is exactly the shadowing and nothing else. The worker
    imports nothing local, so it loses nothing.

    Renaming this module would also have worked, at the cost of
    `ran.baselines.omnifold` no longer being called after the thing it runs.
    """
    return os.environ | {"PYTHONSAFEPATH": "1"}


def _invoke(worker: Path, in_path: Path, out_path: Path) -> None:
    """Run the worker, translating the two failures a caller can act on."""
    # Fixed argv, no shell; the interpolated elements are paths this process
    # created. `--no-project` is essential rather than defensive: without it uv
    # resolves the script against this project, whose `requires-python` is
    # `>=3.14` and cannot be reconciled with the worker's `==3.13.*`.
    command: list[str] = [
        "uv",
        "run",
        "--no-project",
        str(worker),
        str(in_path),
        str(out_path),
    ]
    try:
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_SECONDS,
            env=_worker_env(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "`uv` was not found on PATH. The OmniFold baseline runs its worker "
            "through `uv run`, which provisions the Python 3.13 and TensorFlow "
            "environment this project cannot contain. Install uv, or run the "
            "baseline from a checkout that has it."
        ) from error

    if completed.returncode != 0:
        raise RuntimeError(
            f"The OmniFold worker exited {completed.returncode}.\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )


def _warn_if_on_cpu(device: str) -> None:
    """Say so, loudly, when the worker did not get a GPU.

    This is a warning rather than an error on purpose: the weights are correct
    and a CPU run of a small configuration is a legitimate thing to want. What is
    not legitimate is not knowing, which is what happens by default --- TF logs
    its CUDA troubles and carries on.
    """
    if "GPU" in device:
        logger.info("OmniFold worker ran on %s", device)
        return
    logger.warning(
        "OmniFold worker ran on %s, not a GPU. TensorFlow does not raise when "
        "it cannot load its CUDA libraries, it just runs slowly. On Perlmutter "
        "this is what `module load cudatoolkit/12.9` fixes; see "
        "benchmarks/gpu_coexistence.py.",
        device,
    )


def unfold(
    x_data: EventArray,
    x_sim: EventArray,
    z_gen: EventArray,
    z_target: EventArray,
    out_dir: Path,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    worker: Path | None = None,
) -> EventArray:
    """Per-event weights for `z_target`, mean one, from OmniFold.

    `worker` overrides the packaged script, which is how the tests exercise this
    seam without a TensorFlow environment. Nothing in the package passes it.
    """
    with worker_script() as packaged:
        script: Path = worker if worker is not None else packaged
        with tempfile.TemporaryDirectory() as tmp:
            in_path: Path = Path(tmp) / "in.npz"
            out_path: Path = Path(tmp) / "out.npz"
            np.savez(
                in_path,
                x_data=x_data,
                x_sim=x_sim,
                z_gen=z_gen,
                z_target=z_target,
                niter=np.array(n_iterations),
                epochs=np.array(n_epochs),
                batch_size=np.array(batch_size),
                out_dir=np.array(str(out_dir)),
            )
            _invoke(script, in_path, out_path)
            with np.load(file=out_path, allow_pickle=False) as handle:
                # Same narrowing `data/jets.py` uses: an NpzFile's `__getitem__`
                # carries no element type, and every reader here needs one.
                result: Mapping[str, NDArray[Any]] = cast(
                    "Mapping[str, NDArray[Any]]", handle
                )
                weights: EventArray = np.asarray(a=result["weights"])
                device: str = str(result["device"]) if "device" in handle else "unknown"
                _record_worker_timings(result)

    _warn_if_on_cpu(device)
    return weights


# The worker's own phases, in the order it runs them, with what each covers.
_WORKER_PHASES: tuple[tuple[str, str], ...] = (
    ("init_seconds", "DataLoaders and the two MLPs"),
    ("unfold_seconds", "MultiFold.Unfold"),
    ("reweight_seconds", "evaluating weights on z_target"),
)


def _record_worker_timings(result: Mapping[str, NDArray[Any]], /) -> None:
    """Fold the worker's breakdown into this run's timing tree.

    These are measured in another interpreter, under another Python, so there
    is no block here to wrap and `timing.record` is what puts them in. Called
    from inside `with timing.phase("omnifold")`, so they nest under it the way
    a local sub-phase would.

    The per-iteration step rows go a level deeper still. They are the useful
    part of the breakdown: MultiFold's two steps are not symmetric --- step 1
    reweights at detector level, step 2 at particle level --- so a single
    `unfold` total cannot say which half a long run spent its time in, nor
    whether the cost per iteration is flat or climbing.
    """
    for key, detail in _WORKER_PHASES:
        if key not in result:
            continue
        timing.record(key.removesuffix("_seconds"), float(result[key]), detail=detail)
        # Immediately after `unfold`, because they are its breakdown and the
        # table is read in order.
        if key == "unfold_seconds":
            _record_iteration_timings(result)

    logged: list[str] = [
        f"{key.removesuffix('_seconds')}={float(result[key]):.1f}s"
        for key, _ in _WORKER_PHASES
        if key in result
    ]
    if logged:
        logger.info("OmniFold worker timings: %s", ", ".join(logged))


def _record_iteration_timings(result: Mapping[str, NDArray[Any]], /) -> None:
    """One row per MultiFold iteration per step, beside `unfold` rather than in it.

    They belong *under* `unfold` and are recorded at the same depth anyway,
    because `timing`'s tree is only one level deep in practice: `_ordered`
    reconstructs a top-level phase's children by position and does not recurse,
    so a genuine grandchild renders under whichever sibling happens to precede
    it, and its own parent row prints after it. Rather than rework that for one
    baseline, these sit as siblings of `unfold` in the order they happened,
    which reads correctly and stays honest about the nesting the format
    supports.

    Absent when the wrapping in the worker found nothing to wrap, which is how
    a rename inside OmniFold degrades: the totals still arrive.
    """
    for key, step, what in (
        ("step1_seconds", 1, "detector-level reweighting"),
        ("step2_seconds", 2, "particle-level reweighting"),
    ):
        if key not in result:
            continue
        for iteration, seconds in enumerate(np.atleast_1d(result[key]), start=1):
            timing.record(f"iter{iteration}_step{step}", float(seconds), detail=what)


def _metrics_for(
    config: RunConfig, data: UnfoldingPopulations, weights: EventArray
) -> dict[str, MetricRecord]:
    """Both levels, scored by the same helpers every other baseline uses."""
    metrics: dict[str, MetricRecord] = {}
    for dimension, name in enumerate(iterable=config.variable_names):
        metrics[f"detector_{name}"] = evaluate_dimension(
            data.test.data[:, dimension], data.test.mc.x[:, dimension], weights
        )
        metrics[f"particle_{name}"] = evaluate_dimension(
            data.test.truth[:, dimension], data.test.mc.z[:, dimension], weights
        )
    return metrics


def evaluate_single(
    run_dir: Path,
    force: bool = False,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, MetricRecord]:
    """Run OmniFold on one run's dataset and save its comparison metrics."""
    out_path: Path = artifacts_dir(run_dir) / "metrics_omnifold.json"
    weights_path: Path = artifacts_dir(run_dir) / "omnifold_weights.npz"

    if out_path.exists() and weights_path.exists() and not force:
        logger.info(
            "%s: metrics_omnifold.json exists, skipping (use --force)", run_dir.name
        )
        return cast("dict[str, MetricRecord]", json.loads(s=out_path.read_text()))

    with timing.phase("parse_config"):
        raw_config: object = json.loads(s=(run_dir / "config.json").read_text())
        config: RunConfig = parse_run_config(raw_config)

    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...",
        run_dir.name,
        n_iterations,
        n_epochs,
    )

    detail: str = f"{config.dataset.value} (n={config.n_samples}, dim={config.dim})"
    with timing.phase("data", detail=detail):
        data: UnfoldingPopulations = load_populations(config)

    # The weights are a function of particle-level MC only, and are evaluated on
    # the test split so nothing the unfolding was fit on is scored.
    with timing.phase("omnifold", detail=f"niter={n_iterations}, epochs={n_epochs}"):
        weights: EventArray = unfold(
            x_data=data.fit.data,
            x_sim=data.fit.mc.x,
            z_gen=data.fit.mc.z,
            z_target=data.test.mc.z,
            out_dir=artifacts_dir(run_dir),
            n_iterations=n_iterations,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

    with timing.phase("evaluate", detail=f"{len(config.variable_names)} variables"):
        metrics: dict[str, MetricRecord] = _metrics_for(config, data, weights)

    json.dump(obj=metrics, fp=out_path.open(mode="w"), indent=2)
    np.savez(weights_path, weights=weights)
    logger.info(
        "%s: saved OmniFold metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [OmniFold]", metrics, list(config.variable_names))

    # Its own file, not `timings.json`. `timing.write` merges by phase name
    # alone, and this pass has phases called `data` and `evaluate` too --
    # writing them into the shared file would silently replace the training
    # pass's rows, which are the ones anyone wants. Separate also keeps the
    # baseline's cost separable from the method's, which is the comparison the
    # numbers exist for.
    timing.report()
    # `pass_name` names which invocation produced each row; bandit's
    # hardcoded-password check matches the "pass" substring, as `pyproject.toml`
    # already records for `tests/test_timing.py`.
    timing.write(
        run_dir,
        pass_name="omnifold",  # ruff: ignore[hardcoded-password-func-arg]
        filename="timings_omnifold.json",
    )
    return metrics


def evaluate_runs(
    run_dir: Path = Path("runs"),
    force: bool = False,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    apply_to_runs(
        run_dir,
        evaluate_one=lambda run_dir: evaluate_single(
            run_dir, force, n_iterations, n_epochs, batch_size
        ),
        description="evaluate with OmniFold",
        log=logger,
    )
