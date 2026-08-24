#!/bin/bash
#SBATCH -qshared
#SBATCH -Cgpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -G1
#SBATCH -c32

#SBATCH -t00:15:00
#SBATCH -Am3246_g
#SBATCH -Jran_e2e
#SBATCH -oslurm-%j.log

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
cd "${PROJECT_DIR}"

echo "RAN_CACHE_DIR = ${RAN_CACHE_DIR:-<unset: using ./.cache>}"
TRAIN_ARGS=(-Djets -n1000000 -l3 -u128 -P100)

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
