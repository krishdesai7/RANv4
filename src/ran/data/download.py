"""One-time download of jet substructure data from Zenodo (record 3548091).

Downloads Pythia26 and Herwig Z+jets Delphes datasets (17 .npz files each),
extracts 6 substructure variables, saves per-variable .npz files to .cache/,
and deletes the raw downloads.
"""

from pathlib import Path
import urllib.request

import numpy as np
import numpy.typing as npt

ZENODO_RECORD = 3548091
GENERATORS = ("Pythia26", "Herwig")
N_FILES = 17
SUBSTRUCTURE_VARIABLES = ("m", "M", "w", "tau21", "zg", "sdm")

# Cache-safe filenames: avoid case collisions on case-insensitive filesystems
# (macOS APFS default), where "m.npz" and "M.npz" resolve to the same path.
CACHE_FILENAMES: dict[str, str] = {
    "m": "mass", "M": "mult", "w": "w", "tau21": "tau21", "zg": "zg", "sdm": "sdm",
}

# Only load the keys we actually need (skip particles, Zs, lhas, ang2s).
_NEEDED_KEYS = frozenset({
    "gen_jets", "sim_jets",
    "gen_mults", "sim_mults",
    "gen_widths", "sim_widths",
    "gen_tau2s", "sim_tau2s",
    "gen_zgs", "sim_zgs",
    "gen_sdms", "sim_sdms",
})


def _download_url(generator: str, file_idx: int) -> str:
    return (
        f"https://zenodo.org/record/{ZENODO_RECORD}/files/"
        f"{generator}_Zjet_pTZ-200GeV_{file_idx}.npz?download=1"
    )


def _download_file(url: str, dest: Path) -> None:
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            mb_done = block_num * block_size / 1_000_000
            mb_total = total_size / 1_000_000
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {dest.name}: {pct:3d}% ({mb_done:.0f}/{mb_total:.0f} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _get_var(data: dict[str, npt.NDArray], var: str, ptype: str) -> npt.NDArray:
    """Extract a substructure variable from raw arrays.

    Ported from legacy jet_data.py.
    """
    if var == "m":
        return data[f"{ptype}_jets"][:, 3]
    elif var == "M":
        return data[f"{ptype}_mults"].astype(np.float64)
    elif var == "w":
        return data[f"{ptype}_widths"]
    elif var == "tau21":
        return data[f"{ptype}_tau2s"] / (data[f"{ptype}_widths"] + 1e-50)
    elif var == "zg":
        return data[f"{ptype}_zgs"]
    elif var == "sdm":
        jet_pt_sq: npt.NDArray = data[f"{ptype}_jets"][:, 0] ** 2
        eps: float = 1e-12 * np.mean(jet_pt_sq)
        return np.log(data[f"{ptype}_sdms"] ** 2 / np.maximum(jet_pt_sq, eps) + eps)
    else:
        raise ValueError(f"Unknown variable '{var}'")


def download_jet_data(cache_dir: Path = Path(".cache")) -> None:
    """Download Pythia26/Herwig data from Zenodo, extract variables, save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, dict[str, npt.NDArray]] = {}
    all_raw_paths: list[Path] = []

    for gen in GENERATORS:
        print(f"\n{'=' * 60}")
        print(f"Downloading {gen} ({N_FILES} files)")
        print(f"{'=' * 60}")

        arrays: dict[str, list[npt.NDArray]] = {}

        for i in range(N_FILES):
            path = cache_dir / f"{gen}_Zjet_pTZ-200GeV_{i}.npz"
            all_raw_paths.append(path)

            if not path.exists():
                _download_file(_download_url(gen, i), path)
            else:
                print(f"  {path.name}: already downloaded")

            with np.load(path) as f:
                for key in _NEEDED_KEYS:
                    if key in f:
                        arrays.setdefault(key, []).append(f[key])

        raw_data[gen] = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
        n_events = len(next(iter(raw_data[gen].values())))
        print(f"  {gen}: {n_events:,} events loaded")

    # Herwig = data (nature), Pythia26 = MC (synthetic)
    nature = raw_data["Herwig"]
    synthetic = raw_data["Pythia26"]

    print(f"\nExtracting substructure variables...")
    for var in SUBSTRUCTURE_VARIABLES:
        out_path = cache_dir / f"{CACHE_FILENAMES[var]}.npz"
        np.savez_compressed(
            out_path,
            z_true=_get_var(nature, var, "gen"),
            x_data=_get_var(nature, var, "sim"),
            z_gen=_get_var(synthetic, var, "gen"),
            x_sim=_get_var(synthetic, var, "sim"),
        )
        print(f"  Saved {out_path}")

    # Clean up raw files
    for path in all_raw_paths:
        if path.exists():
            path.unlink()
    print(f"\nCleaned up {len(all_raw_paths)} raw files.")
    print("Done! Jet data cached.")


if __name__ == "__main__":
    download_jet_data()
