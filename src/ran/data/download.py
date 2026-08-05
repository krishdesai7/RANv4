"""One-time download of jet substructure data from Zenodo (record 3548091).

Downloads Pythia26 and Herwig Z+jets Delphes datasets (17 .npz files each),
extracts 6 substructure variables, saves per-variable .npz files to .cache/,
and deletes the raw downloads.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

ZENODO_RECORD = 3548091
GENERATORS = ("Pythia26", "Herwig")
N_FILES = 17
SUBSTRUCTURE_VARIABLES = ("m", "M", "w", "tau21", "zg", "sdm")

# Cache-safe filenames: avoid case collisions on case-insensitive filesystems
# (macOS APFS default), where "m.npz" and "M.npz" resolve to the same path.
CACHE_FILENAMES: dict[str, str] = {
    "m": "mass",
    "M": "mult",
    "w": "w",
    "tau21": "tau21",
    "zg": "zg",
    "sdm": "sdm",
}

# Only load the keys we actually need (skip particles, Zs, lhas, ang2s).
_NEEDED_KEYS = frozenset(
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
    # ruff: ignore[suspicious-url-open-usage] -- scheme checked immediately above
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    logger.info("Downloaded %s", dest)


def _get_var(data: dict[str, npt.NDArray], var: str, ptype: str) -> npt.NDArray:
    """Extract a substructure variable from raw arrays.

    Ported from legacy jet_data.py.
    """
    if var == "m":
        return data[f"{ptype}_jets"][:, 3]
    if var == "M":
        return data[f"{ptype}_mults"].astype(np.float64)
    if var == "w":
        return data[f"{ptype}_widths"]
    if var == "tau21":
        return data[f"{ptype}_tau2s"] / (data[f"{ptype}_widths"] + 1e-50)
    if var == "zg":
        return data[f"{ptype}_zgs"]
    if var == "sdm":
        jet_pt_sq: npt.NDArray = data[f"{ptype}_jets"][:, 0] ** 2
        eps: float = 1e-12 * np.mean(jet_pt_sq)
        return np.log(data[f"{ptype}_sdms"] ** 2 / np.maximum(jet_pt_sq, eps) + eps)
    raise ValueError(f"Unknown variable '{var}'")


def download_jet_data(cache_dir: Path = Path(".cache")) -> None:
    """Download Pythia26/Herwig data from Zenodo, extract variables, save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, dict[str, npt.NDArray]] = {}
    all_raw_paths: list[Path] = []

    with Progress() as progress:
        for gen in GENERATORS:
            logger.info("Downloading %s (%d files)", gen, N_FILES)

            arrays: dict[str, list[npt.NDArray]] = {}

            for i in range(N_FILES):
                path = cache_dir / f"{gen}_Zjet_pTZ-200GeV_{i}.npz"
                all_raw_paths.append(path)

                if not path.exists():
                    task_id = progress.add_task(path.name, total=None)
                    _download_file(_download_url(gen, i), path, progress, task_id)
                    progress.remove_task(task_id)
                else:
                    logger.info("%s: already downloaded", path.name)

                with np.load(path) as f:
                    for key in _NEEDED_KEYS:
                        if key in f:
                            arrays.setdefault(key, []).append(f[key])

            raw_data[gen] = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
            n_events = len(next(iter(raw_data[gen].values())))
            logger.info("%s: %d events loaded", gen, n_events)

    # Herwig = data (nature), Pythia26 = MC (synthetic)
    nature = raw_data["Herwig"]
    synthetic = raw_data["Pythia26"]

    logger.info("Extracting substructure variables...")
    for var in SUBSTRUCTURE_VARIABLES:
        out_path = cache_dir / f"{CACHE_FILENAMES[var]}.npz"
        np.savez_compressed(
            out_path,
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
    logger.info("Jet data cached.")


if __name__ == "__main__":
    download_jet_data()
