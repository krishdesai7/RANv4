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
    from typing import Any, Final, LiteralString

    from numpy.typing import NDArray

logger: Logger = logging.getLogger(name=__name__)

# Only load the keys we actually need (skip particles, Zs, lhas, ang2s).
_NEEDED_KEYS: frozenset[LiteralString] = frozenset(
    {
        "gen_jets",
        "sim_jets",
        "gen_mults",
        "sim_mults",
        "gen_widths",
        "sim_widths",
        "gen_tau2s",
        "sim_tau2s",
        "gen_zgs",
        "sim_zgs",
        "gen_sdms",
        "sim_sdms",
    }
)


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


def _get_var(data: dict[str, NDArray], var: str, ptype: str, /) -> NDArray[np.double]:
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
    raise ValueError(f"Unknown variable '{var}'")


def _ensure_shard(path: Path, gen: str, idx: int, progress: Progress, /) -> None:
    """Download one shard unless it is already cached."""
    if path.exists():
        logger.info("%s: already downloaded", path.name)
        return
    task_id: TaskID = progress.add_task(description=path.name, total=None)
    _download_file(_download_url(generator=gen, file_idx=idx), path, progress, task_id)
    progress.remove_task(task_id)


def _fetch_generator(
    gen: str, cache_dir: Path, progress: Progress, all_raw_paths: list[Path]
) -> dict[str, NDArray]:
    logger.info("Downloading %s (%d files)", gen, N_FILES)
    arrays: dict[str, list[NDArray]] = {}
    for i in range(N_FILES):
        path: Path = cache_dir / f"{gen}_Zjet_pTZ-200GeV_{i}.npz"
        all_raw_paths.append(path)
        _ensure_shard(path, gen, i, progress)
        with np.load(file=path) as f:
            for key in _NEEDED_KEYS:
                if key in f:
                    arrays.setdefault(key, []).append(f[key])
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def download_jet_data(cache_dir: Path = CACHE_DIR) -> None:
    """Download Pythia26/Herwig data from Zenodo, extract variables, save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, dict[str, NDArray]] = {}
    all_raw_paths: list[Path] = []

    with Progress() as progress:
        for gen in GENERATORS:
            raw_data[gen] = _fetch_generator(gen, cache_dir, progress, all_raw_paths)
            n_events: int = len(next(iter(raw_data[gen].values())))
            logger.info("%s: %d events loaded", gen, n_events)

    # Herwig = data (nature), Pythia26 = MC (synthetic)
    nature: dict[str, NDArray[Any]] = raw_data["Herwig"]
    synthetic: dict[str, NDArray[Any]] = raw_data["Pythia26"]

    logger.info(msg="Extracting substructure variables...")
    for var in SUBSTRUCTURE_VARIABLES:
        out_path: Path = cache_dir / f"{CACHE_FILENAMES[var]}.npz"
        np.savez_compressed(
            file=out_path,
            z_true=_get_var(nature, var, "gen"),
            x_data=_get_var(nature, var, "sim"),
            z_gen=_get_var(synthetic, var, "gen"),
            x_sim=_get_var(synthetic, var, "sim"),
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
