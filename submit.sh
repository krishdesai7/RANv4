#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m3246_g
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log

cd /global/u1/k/kdesai/RANv4
uv run python -m ran "$@"
