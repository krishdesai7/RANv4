#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m3246_g
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log

cd /global/u1/k/kdesai/RANv4
uv run -m ran "$@"

# Find the most recent run and compute OmniFold baseline
LATEST_RUN=$(ls -dt runs/*/ | head -1)
echo "Running OmniFold baseline on ${LATEST_RUN}."
uv run -m ran.baselines.omnifold --run_dir="${LATEST_RUN}"
