"""The `B x S` design: bootstrap datasets crossed with initialization seeds.

One invocation runs one cell, so a cluster can put every cell on its own GPU
and the whole design costs one training run of wall clock. `cell_of_index`
maps a flat array-task id onto `(dataset, seed)`, and each cell writes a small
`.npz` carrying its weights on the common evaluation set --- nothing else, and
in particular no model, because the design is about the spread of the outputs
rather than any one run's parameters.

Three decisions in here are the ones that make the numbers mean what the
report says they mean:

**The bootstrap resamples events, not the split.** Varying `data_seed`
reshuffles a fixed sample into different train/val/test splits and different
batch orders; every run still sees the same 1M events. That is *method*
variance --- an artifact of the algorithm being order-dependent, removable by
ensembling --- and it is not the statistical uncertainty a measurement is
obliged to report. The nonparametric bootstrap, drawing `n` of `n` with
replacement, is what estimates the latter: how much the answer would move if
the experiment had collected a different sample of the same size. `data_seed`
is therefore held **fixed** across the whole design.

**MC and nature resample independently.** They are two separate samples in the
physics --- one generated, one measured --- and coupling their resampling
would impose a correlation that does not exist. The pairing *within* each is
preserved, because `(z_gen, x_sim)` and `(x_data, z_true)` are the same events
seen at two levels.

**Every cell is evaluated on one fixed common set of gen-level events**, held
out before any resampling and therefore absent from every replicate's
training. Bootstrap replicates contain different --- and duplicated --- events,
so their per-event weight vectors are not otherwise commensurable and cannot be
stacked into the matrix the covariance is computed from. Holding the set out
also means the finite size of the evaluation sample shifts every run together
and cancels out of the across-run contrast entirely.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from ..data import RANDataset, load_jet_dataset, parse_gaussian_config
from ..rantypes import (
    SUBSTRUCTURE_VARIABLES,
    DatasetName,
    Events,
    Populations,
    Split,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from ..rantypes import DatasetSplits, EventArray, GaussianConfig

logger: Logger = logging.getLogger(name=__name__)

CELL_GLOB: str = "cell_*.npz"


class DesignSpec(NamedTuple):
    """The shape of the grid, and the seeds every cell derives its own from."""

    n_datasets: int
    n_seeds: int
    data_seed: int = 42
    init_seed: int = 0

    @property
    def n_cells(self) -> int:
        return self.n_datasets * self.n_seeds

    def cell_of_index(self, index: int, /) -> tuple[int, int]:
        """Flat task id -> `(dataset, seed)`, seed-major within a dataset.

        Seed-major so that a design cut short after `k * n_seeds` cells is a
        complete grid over the datasets that finished, rather than a ragged
        one no decomposition will accept.
        """
        if not 0 <= index < self.n_cells:
            raise IndexError(
                f"cell {index} is outside a {self.n_datasets}x{self.n_seeds} "
                f"design ({self.n_cells} cells)"
            )
        return divmod(index, self.n_seeds)


class EvaluationSet(NamedTuple):
    """What is left to train on, and the gen-level events every run is read on."""

    pool: Populations
    z: EventArray


def reserve_evaluation_set(
    pops: Populations, /, *, n_eval: int, seed: int | Sequence[int]
) -> EvaluationSet:
    """Hold out `n_eval` MC gen-level events, before any resampling touches them.

    Only the MC side is reserved: `g` is evaluated on `z_gen` and never on a
    nature event, so there is nothing to hold out on that side and no reason
    to spend the statistics.
    """
    n_mc: int = len(pops.mc)
    if not 0 < n_eval < n_mc:
        raise ValueError(
            f"n_eval must be between 1 and {n_mc - 1} (the MC events available), "
            f"got {n_eval}"
        )
    order: NDArray[np.intp] = np.random.default_rng(seed).permutation(x=n_mc)
    held, kept = order[:n_eval], order[n_eval:]
    return EvaluationSet(
        pool=Populations(
            mc=Events(z=pops.mc.z[kept], x=pops.mc.x[kept]),
            data=pops.data,
            truth=pops.truth,
        ),
        z=pops.mc.z[held],
    )


def bootstrap(pops: Populations, /, *, seed: int | Sequence[int]) -> Populations:
    """One nonparametric bootstrap replicate: `n` of `n` with replacement.

    Both samples keep their original size, so a replicate is the same
    measurement repeated rather than a smaller one, and the variance it
    estimates is the variance at the size actually collected.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n_mc: int = len(pops.mc)
    n_nature: int = pops.data.shape[0]
    i_mc: NDArray[np.intp] = rng.integers(low=0, high=n_mc, size=n_mc)
    i_nature: NDArray[np.intp] = rng.integers(low=0, high=n_nature, size=n_nature)
    return Populations(
        mc=Events(z=pops.mc.z[i_mc], x=pops.mc.x[i_mc]),
        data=pops.data[i_nature],
        truth=pops.truth[i_nature],
    )


def base_populations(
    dataset: DatasetName,
    /,
    *,
    n_samples: int,
    batch_size: int,
    data_seed: int,
    variables: tuple[str, ...] = SUBSTRUCTURE_VARIABLES,
    params: GaussianConfig | None = None,
) -> tuple[Populations, int]:
    """The undisturbed sample the design bootstraps, plus its dimensionality.

    Built through the ordinary loaders and immediately flattened back out of
    the splits: the design does its own splitting per replicate, so the split
    boundaries here would only be re-drawn.

    The Gaussian branch takes already-parsed parameters rather than a YAML
    path, so `collect` can rebuild the identical sample months later from what
    the cells recorded, without the file still having to be on disk and
    unchanged.
    """
    if dataset == DatasetName.jets:
        splits: DatasetSplits = load_jet_dataset(
            n_samples=n_samples,
            batch_size=batch_size,
            variables=variables,
            seed=data_seed,
        )[0]
        return splits.select(Split.ALL).partition(), len(variables)
    if dataset == DatasetName.gaussian:
        if params is None:
            raise ValueError("Gaussian mode requires --config path/to/config.yaml")
        splits = RANDataset(batch_size, data_seed).generate_gaussian_dataset(
            params=params, n_samples=n_samples
        )
        return splits.select(Split.ALL).partition(), params.dim
    raise ValueError(f"Unknown dataset: {dataset!r}")


def cell_path(design_dir: Path, index: int, /) -> Path:
    return design_dir / f"cell_{index:04d}.npz"


def run_cell(
    index: int,
    design_dir: Path,
    spec: DesignSpec,
    /,
    *,
    dataset: DatasetName = DatasetName.jets,
    variables: tuple[str, ...] = SUBSTRUCTURE_VARIABLES,
    config: Path | None = None,
    n_samples: int = 500_000,
    n_eval: int = 100_000,
    batch_size: int = 1024,
    hidden_units: int = 64,
    n_layers: int = 2,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 3e-5,
    lr_d: float = 1e-4,
    lambda_dispersion: float = 0.015,
) -> Path:
    """Train one `(dataset, seed)` cell and record its weights on the common set."""
    # Deferred so that `ran uncertainty collect`, which only reads npz and
    # reports, does not pay for importing keras and jax.
    from ..evaluate import _get_weights
    from ..train import train

    b, s = spec.cell_of_index(index)
    params: GaussianConfig | None = (
        parse_gaussian_config(config)
        if dataset == DatasetName.gaussian and config is not None
        else None
    )
    pops, dim = base_populations(
        dataset,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=spec.data_seed,
        variables=variables,
        params=params,
    )
    # Seeded off `spec.data_seed` alone, with no `b`: every cell has to reserve
    # the *same* events or the weight vectors are not stackable.
    evaluation: EvaluationSet = reserve_evaluation_set(
        pops, n_eval=n_eval, seed=spec.data_seed
    )
    # Two ints rather than an arithmetic combination, so no two replicates can
    # collide on one stream however the base seed is chosen.
    replicate: Populations = bootstrap(evaluation.pool, seed=(spec.data_seed, b))

    splits: DatasetSplits = RANDataset(
        batch_size=batch_size, seed=spec.data_seed
    ).splits_from_data(replicate.interleave())
    result = train(
        splits,
        dim,
        hidden_units,
        n_layers,
        spec.init_seed + s,
        n_epochs=n_epochs,
        n_disc_steps=n_disc_steps,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_dispersion=lambda_dispersion,
    )
    weights: EventArray = _get_weights(result.g, evaluation.z)

    design_dir.mkdir(parents=True, exist_ok=True)
    out: Path = cell_path(design_dir, index)
    np.savez(
        file=out,
        weights=weights,
        meta=np.array(
            object=json.dumps(
                obj={
                    "index": index,
                    "dataset_index": b,
                    "seed_index": s,
                    "init_seed": result.seed,
                    "data_seed": spec.data_seed,
                    "n_eval": n_eval,
                    "n_samples": n_samples,
                    "batch_size": batch_size,
                    "dim": dim,
                    "dataset": dataset.value,
                    "variables": list(variables),
                    "gaussian_params": params.model_dump() if params else None,
                    "mmd_test": result.mmd_test,
                }
            )
        ),
    )
    logger.info(
        "cell %d (dataset %d, seed %d): test MMD^2 %.3e -> %s",
        index,
        b,
        s,
        result.mmd_test,
        out,
    )
    return out


class Design(NamedTuple):
    """A loaded grid: `(B, S, n_eval)` weights plus what rebuilds its inputs.

    `meta` is cell zero's record with the per-cell fields dropped, because the
    rest of it --- dataset, variables, sample size, seeds --- is by
    construction identical across the grid, and is what `collect` needs to
    regenerate the common evaluation set without storing a copy of it in every
    cell.
    """

    weights: NDArray[np.double]
    spec: DesignSpec
    meta: dict[str, Any]


_PER_CELL_KEYS: frozenset[str] = frozenset(
    ("index", "dataset_index", "seed_index", "init_seed", "mmd_test")
)


def _read_cell(path: Path, /) -> tuple[NDArray[np.double], dict[str, Any]]:
    with np.load(file=path) as cell:
        arrays: Mapping[str, NDArray[Any]] = cast("Mapping[str, NDArray[Any]]", cell)
        return (
            np.asarray(a=arrays["weights"], dtype=np.double),
            json.loads(s=str(object=arrays["meta"].item())),
        )


def _missing_cells(design_dir: Path, spec: DesignSpec, /) -> list[int]:
    return [
        index
        for index in range(spec.n_cells)
        if not cell_path(design_dir, index).exists()
    ]


def load_cells(design_dir: Path, spec: DesignSpec, /) -> Design:
    """Stack the finished cells into a `(B, S, n_eval)` grid, or say what is missing.

    The decomposition needs a *balanced* grid --- the mean squares assume one
    run per cell --- so a partial design is refused with the list of gaps
    rather than silently decomposed over whatever landed, which would charge
    the imbalance to the dataset axis.
    """
    missing: list[int] = _missing_cells(design_dir, spec)
    if missing:
        raise FileNotFoundError(
            f"{design_dir} is missing cells {missing}; the decomposition needs "
            f"all {spec.n_cells} of a {spec.n_datasets}x{spec.n_seeds} grid"
        )

    first, meta = _read_cell(cell_path(design_dir, 0))
    grid: NDArray[np.double] = np.empty(
        shape=(spec.n_datasets, spec.n_seeds, first.size)
    )
    for index in range(spec.n_cells):
        weights, _ = _read_cell(cell_path(design_dir, index))
        if weights.size != first.size:
            raise ValueError(
                f"cell {index} holds {weights.size} weights but cell 0 holds "
                f"{first.size}; these cells are not from one design"
            )
        b, s = spec.cell_of_index(index)
        grid[b, s] = weights
    return Design(
        weights=grid,
        spec=spec,
        meta={k: v for k, v in meta.items() if k not in _PER_CELL_KEYS},
    )
