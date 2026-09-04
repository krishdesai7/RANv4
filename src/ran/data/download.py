from __future__ import annotations

import logging
import urllib.request
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import Progress, TaskID

from ..rantypes import (
    CACHE_DIR,
    CACHE_FILENAMES,
    GENERATORS,
    LOG_RHO_FLOOR,
    N_FILES,
    SUBSTRUCTURE_VARIABLES,
    ZENODO_RECORD,
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from typing import Final

    from numpy.typing import NDArray

logger: Logger = logging.getLogger(name=__name__)

PID_CHARGE: Final[np.uintc] = np.uintc(0x5228849)


def _download_url(generator: str, file_idx: int) -> str:
    return (
        f"https://zenodo.org/record/{ZENODO_RECORD}/files/"
        f"{generator}_Zjet_pTZ-200GeV_{file_idx}.npz?download=1"
    )


def _download_file(url: str, dest: Path, progress: Progress, task_id: TaskID) -> None:
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            progress.update(
                task_id,
                total=total_size,
                completed=min(block_num * block_size, total_size),
            )

    # urlretrieve honours file:// and any other scheme urllib knows, so pin it to
    # https before opening. Callers only ever pass _download_url output, which is
    # built from a hardcoded https://zenodo.org prefix; this keeps that true if
    # the helper ever grows another caller.
    if not url.startswith("https://"):
        raise ValueError(f"refusing to fetch non-https URL: {url!r}")
    # ruff: ignore[suspicious-url-open-usage]
    urllib.request.urlretrieve(url, filename=dest, reporthook=_progress)
    logger.info("Downloaded %s", dest)


_ONE_PRONG_TAU21: Final[float] = 0.0


# `PID_CHARGE` packs one 2-bit charge field per particle-type index into a
# 32-bit word, so only indices 0..15 are addressable. A larger index shifts by
# 32 or more, which is undefined rather than merely wrong, and would hand back
# a plausible-looking charge for a particle type this table does not cover.
_MAX_PID_INDEX: Final[int] = 15


def _constituents(
    data: dict[str, NDArray], ptype: str, /
) -> tuple[NDArray[np.double], NDArray[np.int8]]:
    """Per-constituent `(pt, charge)` for every jet.

    `particles` is `(jets, constituents, 4)` with columns `(pt, y, phi, pid/10)`
    and the constituent axis zero-padded. Indexing it as `[:, 0]` selects one
    *constituent's* four features rather than every constituent's pt, which is
    a silent axis error whenever a jet happens to have four constituents and a
    broadcast failure otherwise.
    """
    particles: NDArray = data[f"{ptype}_particles"]
    if particles.ndim != 3 or particles.shape[2] < 4:
        raise ValueError(
            f"{ptype}_particles has shape {particles.shape}; "
            "expected (jets, constituents, >=4)"
        )
    pt: NDArray[np.double] = particles[:, :, 0].astype(dtype=np.double)
    # Widened before the shift: `ubyte * 2` wraps at 128, and the index feeds a
    # shift distance rather than a value, so a wrap is silent corruption.
    ids: NDArray[np.intp] = np.round(a=particles[:, :, 3] * 10).astype(dtype=np.intp)
    if ids.min() < 0 or ids.max() > _MAX_PID_INDEX:
        raise ValueError(
            f"{ptype}_particles carries pid indices in "
            f"[{ids.min()}, {ids.max()}]; PID_CHARGE only encodes "
            f"[0, {_MAX_PID_INDEX}]"
        )
    charge: NDArray[np.int8] = (((PID_CHARGE >> (ids * 2)) & 3) - 1).astype(
        dtype=np.int8
    )
    return pt, charge


def _safe_ratio(
    numerator: NDArray[np.double], denominator: NDArray[np.double], /
) -> NDArray[np.double]:
    """`numerator / denominator`, zero where the jet carries no pt at all.

    Same `where=` discipline as the tau21 and sdm branches: an empty jet is a
    degenerate input to guard explicitly, not a nan to propagate into the
    standardization statistics of every other event.
    """
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(shape=denominator.shape, dtype=np.double),
        where=denominator > 0.0,
    )


# Derived from the per-constituent `particles` array rather than read from a
# stored per-jet array. Kept as a set so `_get_var` dispatches once instead of
# falling through one branch per observable.
_CONSTITUENT_VARS: Final[frozenset[str]] = frozenset({"q", "f_ch", "n_ch", "ptd"})


def _stored_var(
    data: dict[str, NDArray], var: str, ptype: str, /
) -> NDArray[np.double]:
    """Observables the Zenodo release already carries, one value per jet."""
    match var:
        case "m":
            return data[f"{ptype}_jets"][:, 3].astype(dtype=np.double)
        case "M":
            return data[f"{ptype}_mults"].astype(dtype=np.double)
        case "w":
            return data[f"{ptype}_widths"].astype(dtype=np.double)
        case "tau21":
            tau2: NDArray[np.double] = data[f"{ptype}_tau2s"].astype(dtype=np.double)
            width: NDArray[np.double] = data[f"{ptype}_widths"].astype(dtype=np.double)
            return np.divide(
                tau2,
                width,
                out=np.full(shape=width.shape, fill_value=_ONE_PRONG_TAU21),
                where=width > 0,
            )
        case "zg":
            return data[f"{ptype}_zgs"].astype(dtype=np.double)
        case "sdm":
            sdm: NDArray[np.double] = data[f"{ptype}_sdms"].astype(dtype=np.double)
            jet_pt: NDArray[np.double] = data[f"{ptype}_jets"][:, 0].astype(
                dtype=np.double
            )
            rho_sq: NDArray[np.double] = (sdm / jet_pt) ** 2
            return np.log(
                rho_sq,
                out=np.full(shape=rho_sq.shape, fill_value=LOG_RHO_FLOOR),
                where=rho_sq > 0,
            )
        case "lha":
            return data[f"{ptype}_lhas"].astype(dtype=np.double)
        case "ang2":
            return data[f"{ptype}_ang2s"].astype(dtype=np.double)
    raise ValueError(f"Unknown variable '{var}'")


def _constituent_var(
    data: dict[str, NDArray], var: str, ptype: str, /
) -> NDArray[np.double]:
    """Observables computed from the jet's constituents."""
    pt, charge = _constituents(data, ptype)
    match var:
        case "q":
            # Jet charge, pT^kappa-weighted at kappa = 1/2.
            root_pt: NDArray[np.double] = np.sqrt(pt)
            return _safe_ratio((charge * root_pt).sum(axis=1), root_pt.sum(axis=1))
        case "f_ch":
            return _safe_ratio((np.abs(charge) * pt).sum(axis=1), pt.sum(axis=1))
        case "n_ch":
            # Masked on pt, not on charge alone. The constituent axis is
            # zero-padded to the longest jet in the file, and a count is the
            # one observable here that a padded row could enter -- every other
            # one weights by pt, which is zero on padding.
            return np.count_nonzero((charge != 0) & (pt > 0.0), axis=1).astype(
                dtype=np.double
            )
        case "ptd":
            # sqrt(sum pt^2) / sum pt: 1 for a one-particle jet, 1/sqrt(n) for
            # n equal ones. Padding contributes zero to both sums.
            return _safe_ratio(np.sqrt(np.square(pt).sum(axis=1)), pt.sum(axis=1))

    raise ValueError(f"Unknown variable '{var}'")


def _get_var(data: dict[str, NDArray], var: str, ptype: str, /) -> NDArray[np.double]:
    if var in _CONSTITUENT_VARS:
        return _constituent_var(data, var, ptype)
    return _stored_var(data, var, ptype)


def _ensure_shard(dest: Path, gen: str, idx: int, progress: Progress, /) -> None:
    """Download one shard unless it is already cached."""
    if dest.exists():
        logger.info("%s: already downloaded", dest.name)
        return
    task_id: TaskID = progress.add_task(description=dest.name, total=None)
    _download_file(_download_url(generator=gen, file_idx=idx), dest, progress, task_id)
    progress.remove_task(task_id)


_COLUMN_KEYS: Final[tuple[str, ...]] = tuple(
    f"{ptype}_{var}" for var in SUBSTRUCTURE_VARIABLES for ptype in ("gen", "sim")
)


def _shard_observables(path: Path, /) -> dict[str, NDArray[np.double]]:
    """Every observable, both levels, for one downloaded shard."""
    # Materialized once: an `NpzFile` decompresses on every member access, and
    # four observables read `particles`.
    with np.load(file=path) as handle:
        shard: dict[str, NDArray] = {key: handle[key] for key in handle.files}
    return {
        f"{ptype}_{var}": _get_var(shard, var, ptype)
        for var in SUBSTRUCTURE_VARIABLES
        for ptype in ("gen", "sim")
    }


def _fetch_generator(
    gen: str,
    cache_dir: Path,
    progress: Progress,
    all_raw_paths: list[Path],
) -> dict[str, NDArray[np.double]]:
    """One generator's observables, reduced shard by shard.

    Returns `{"<ptype>_<var>": values}` over every event, never the raw arrays.

    Reducing inside the loop rather than concatenating first is what makes
    `particles` affordable and correct. Affordable: the constituent arrays are
    the bulk of the release, and holding both generators' at float64 would be
    ~12 GB against ~100 MB of derived observables. Correct: the constituent
    axis is padded to the longest jet *in that array*, which differs between
    shards and even between `gen` and `sim` in one shard (116 against 94 in
    file 0), so a buffer preallocated from the first shard cannot hold the
    rest -- it raises on the first wider one.

    Concatenating the reductions also drops the assumption that every shard
    carries the same number of events.
    """
    logger.info("Downloading %s (%d files)", gen, N_FILES)

    files: list[Path] = [
        cache_dir / f"{gen}_Zjet_pTZ-200GeV_{i}.npz" for i in range(N_FILES)
    ]
    for i, path in enumerate(iterable=files):
        all_raw_paths.append(path)
        _ensure_shard(path, gen, i, progress)

    columns: dict[str, list[NDArray[np.double]]] = {key: [] for key in _COLUMN_KEYS}
    for path in files:
        for key, values in _shard_observables(path).items():
            columns[key].append(values)

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def download_jet_data(cache_dir: Path = CACHE_DIR) -> None:
    """Download Pythia26/Herwig data from Zenodo, extract variables, save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, dict[str, NDArray[np.double]]] = {}
    all_raw_paths: list[Path] = []

    with Progress() as progress:
        for gen in GENERATORS:
            raw_data[gen] = _fetch_generator(gen, cache_dir, progress, all_raw_paths)
            n_events: int = len(next(iter(raw_data[gen].values())))
            logger.info("%s: %d events loaded", gen, n_events)

    # Herwig = data (nature), Pythia26 = MC (synthetic)
    nature: dict[str, NDArray[np.double]] = raw_data["Herwig"]
    synthetic: dict[str, NDArray[np.double]] = raw_data["Pythia26"]

    logger.info(msg="Extracting substructure variables...")
    for var in SUBSTRUCTURE_VARIABLES:
        out_path: Path = cache_dir / f"{CACHE_FILENAMES[var]}.npz"
        # `savez`, not `savez_compressed`. These observables are float64 and
        # very nearly incompressible: measured on representative data, DEFLATE
        # lands at ~0.94 of raw for every continuous variable (mass, w, tau21,
        # zg, sdm) while costing ~20x on read. `mult` is the sole exception at
        # ~0.18, because it is integer-valued -- one variable in six does not
        # pay for the tax on the other five, and the read cost is paid on every
        # run while the write happens once.
        #
        # This is a write-side change only. `np.load` reads stored and deflated
        # members identically, so caches written before this keep working and
        # nothing needs invalidating.
        np.savez(
            file=out_path,
            z_true=nature[f"gen_{var}"],
            x_data=nature[f"sim_{var}"],
            z_gen=synthetic[f"gen_{var}"],
            x_sim=synthetic[f"sim_{var}"],
        )
        logger.info("Saved %s", out_path)

    # Clean up raw files
    for path in all_raw_paths:
        if path.exists():
            path.unlink()
    logger.info("Cleaned up %d raw files.", len(all_raw_paths))
    logger.info(msg="Jet data cached.")


if __name__ == "__main__":
    download_jet_data()
