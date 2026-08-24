#!/bin/bash
# End-to-end unfolding run: train -> IBU baseline -> replot with the baseline
# overlaid -> metrics. Defaults to all six jet observables, which is the
# full-fledged case; every stage runs in one sequential job on one GPU.
#
#   sbatch scripts/submit.sh                       # all 6 jet observables
#   sbatch scripts/submit.sh --seed 7              # extra flags reach `ran train`
#   sbatch scripts/submit.sh --var m --var w       # a subset
#   sbatch scripts/submit.sh --dataset gaussian --config params/2d_correlated.yaml
#
# The last form still works because `--dataset jets` is *prepended*: click keeps
# the last occurrence of a scalar option, so anything on the command line wins.
# `--var` is the exception --- it is `multiple=True` and accumulates --- which is
# why the six-variable default comes from `ran train` itself (an empty `--var`
# means SUBSTRUCTURE_VARIABLES) rather than being spelled out here.
#SBATCH --qos=regular
#SBATCH --constraint=gpu
# One GPU, not four. Nothing in `ran train` shards across devices --- it is a
# single jitted program on device 0 --- so the other three would sit idle while
# JAX preallocated ~75% of each. Asking for one also lands a better queue slot.
#SBATCH --gpus=1
#SBATCH --account=m3246_g
#SBATCH --time=02:00:00
#SBATCH --job-name=ran_e2e
#SBATCH --output=slurm-%j.log

# `set -e` matters more here than in a one-liner: this is four stages, and a
# failed train must not go on to run IBU against a run directory from last week.
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
cd "${PROJECT_DIR}"

# RAN_CACHE_DIR is inherited from the submitting shell (SLURM exports the
# environment by default). It is deliberately not set here: relocating the cache
# re-downloads ~2GB of Zenodo jet data on the first run, which should be a
# considered choice rather than a side effect of running this script.
echo "RAN_CACHE_DIR = ${RAN_CACHE_DIR:-<unset: using ./.cache>}"

# `_save_run` names the run directory for the UTC timestamp and returns it only
# to its Python caller, so a shell has to find it. Anchor on a marker file
# rather than "newest directory": `runs/` already holds older runs, and a
# mistake here would silently attach IBU to one of them.
mkdir -p runs
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT

uv run ran train --dataset jets "$@"

RUN_DIR="$(find runs -mindepth 1 -maxdepth 1 -type d -newer "${marker}" | sort | tail -1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "No run directory appeared under runs/ --- training did not save." >&2
  exit 1
fi
echo "Run dir: ${RUN_DIR}"

# IBU unfolds the same populations for comparison. The reload pass after it is
# what puts the baseline into the figures: `workflow.run` reads ibu_weights.npz
# only if it exists when the plots are drawn, and on the training pass it does
# not exist yet.
uv run ran baseline ibu --run-dir "${RUN_DIR}"
uv run ran train --load-run "${RUN_DIR}"

# The reload pass leaves metrics.json alone (it only forces on a fresh train),
# so recompute explicitly --- this is the number the run exists to produce.
uv run ran evaluate --run-dir "${RUN_DIR}" --force

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"

# Evidence that the persistent compilation cache did its job. A populated
# directory here is what makes the *next* run skip ~4.6s of XLA; an empty one
# means RAN_CACHE_DIR points somewhere unwritable and JAX only warned about it.
jax_cache="${RAN_CACHE_DIR:-.cache}/jax"
echo "XLA cache: $(find "${jax_cache}" -type f 2>/dev/null | wc -l | tr -d ' ') entries in ${jax_cache}"
