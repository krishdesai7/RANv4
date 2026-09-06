#!/bin/bash
#SBATCH -qshared
#SBATCH -Cgpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -G1
#SBATCH -c32

#SBATCH -t00:45:00
#SBATCH -Am3246_g
#SBATCH -Jran_e2e
#SBATCH -oslurm-%j.log

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
cd "${PROJECT_DIR}"

echo "RAN_CACHE_DIR = ${RAN_CACHE_DIR:-<unset: using ./.cache>}"

# Report where the wall clock went. `timings.json` lands in the run directory
# alongside config.json, so it joins against it without walking a tree.
export RAN_TIMING=1

# The full 12: the OmniFold six plus q, f_ch, lha, ang2, ptd, n_ch.
#
# Deliberately NOT passed as `--var` flags. `--var` is a repeatable option, so
# click *appends* rather than replacing -- naming all twelve here would turn a
# `sbatch scripts/submit.sh --var m` into thirteen names with a duplicate, and
# `load_jet_dataset` rejects duplicates. Omitting it lets the loader's own
# default (all of SUBSTRUCTURE_VARIABLES) stand while keeping a subset
# selectable from the command line. Scalar flags below still follow the
# last-occurrence rule, so `-n`/`-u`/`-l` remain overridable.
N_REQUESTED=1600000

# The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset`
# raises rather than truncating when asked for more than is on disk -- which
# would burn the whole allocation on an immediate ValueError. Read the real
# count off the cache and clamp. All twelve observables are derived from the
# same events in one pass of `download_jet_data`, so variables[0] settles it,
# which is the file the loader checks too.
N_SAMPLES="$(uv run python - "${N_REQUESTED}" <<'PY'
import sys

import numpy as np

from ran.rantypes.constants import (
    CACHE_DIR,
    CACHE_FILENAMES,
    SUBSTRUCTURE_VARIABLES,
)

requested = int(sys.argv[1])
missing = [
    v for v in SUBSTRUCTURE_VARIABLES
    if not (CACHE_DIR / f"{CACHE_FILENAMES[v]}.npz").exists()
]
if missing:
    # Cold cache: 3.3GB from Zenodo. Say so and let the run decide -- the job
    # will pull it, which is exactly what warming on a login node avoids.
    print(f"COLD {' '.join(missing)}", file=sys.stderr)
    print(requested)
    raise SystemExit(0)

path = CACHE_DIR / f"{CACHE_FILENAMES[SUBSTRUCTURE_VARIABLES[0]]}.npz"
with np.load(file=path) as f:
    available = min(len(f["z_true"]), len(f["z_gen"]))
print(min(requested, available))
PY
)"

if [[ "${N_SAMPLES}" -lt "${N_REQUESTED}" ]]; then
  echo "Clamped n_samples ${N_REQUESTED} -> ${N_SAMPLES} (all that is on disk)."
fi
echo "n_samples = ${N_SAMPLES}, all 12 jet observables"

TRAIN_ARGS=(-Djets "-n${N_SAMPLES}" -l3 -u128)

mkdir -p runs
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT

uv run ran train "${TRAIN_ARGS[@]}" "$@"

RUN_DIR="$(find runs -mindepth 1 -maxdepth 1 -type d -newer "${marker}" | sort | tail -1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "No run directory appeared under runs/. Training did not save." >&2
  exit 1
fi
echo "Run dir: ${RUN_DIR}"

uv run ran baseline ibu --run-dir "${RUN_DIR}"
uv run ran train --load-run "${RUN_DIR}"
uv run ran evaluate --run-dir "${RUN_DIR}" --force

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"

jax_cache="${RAN_CACHE_DIR:-.cache}/jax"
echo "XLA cache: $(find "${jax_cache}" -type f 2>/dev/null | wc -l | tr -d ' ') entries in ${jax_cache}"
