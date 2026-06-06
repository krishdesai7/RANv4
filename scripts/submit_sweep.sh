#!/bin/bash
# Launch the cubic-response RAN-vs-OmniFold sweep as a SLURM array job.
# Run on the login node:  bash scripts/submit_sweep.sh
# (Do NOT sbatch this file itself; it submits the array + collect jobs.)
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
N_POINTS=25
SWEEP_DIR="${PROJECT_DIR}/runs/cubic_sweep_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${SWEEP_DIR}"
echo "Sweep dir: ${SWEEP_DIR}"

# One array task per s value (indices 0..N_POINTS-1). Each trains RAN + OmniFold
# at its s and writes s_<index>.json into the shared sweep dir.
ARRAY_JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --gpus=1 --account=m3246_g --time=02:00:00 \
  --array=0-$((N_POINTS - 1)) \
  --output="${SWEEP_DIR}/slurm-%A_%a.log" \
  --wrap="cd ${PROJECT_DIR} && uv run -m ran.experiments.cubic_sweep run_point --s_index=\${SLURM_ARRAY_TASK_ID} --sweep_dir=${SWEEP_DIR} --n_points=${N_POINTS}")
echo "Array job: ${ARRAY_JOB}"

# Collect after all array elements finish (afterany: run even if some failed,
# so a single bad point doesn't sink the whole figure).
COLLECT_JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --gpus=1 --account=m3246_g --time=00:20:00 \
  --dependency=afterany:"${ARRAY_JOB}" \
  --output="${SWEEP_DIR}/slurm-collect-%j.log" \
  --wrap="cd ${PROJECT_DIR} && uv run -m ran.experiments.cubic_sweep collect --sweep_dir=${SWEEP_DIR} --n_points=${N_POINTS}")
echo "Collect job: ${COLLECT_JOB} (afterany:${ARRAY_JOB})"
echo "Results will land in ${SWEEP_DIR}/ (results.npz, wasserstein_vs_s.pdf)"
