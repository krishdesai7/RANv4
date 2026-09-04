#!/bin/bash
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
N_POINTS=24

NODES=${NODES:-6}
GPUS_PER_NODE=4
GPUS_TOTAL=$((NODES * GPUS_PER_NODE))

SWEEP_DIR="${PROJECT_DIR}/runs/cubic_sweep_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${SWEEP_DIR}"
echo "Sweep dir: ${SWEEP_DIR}"
echo "Requesting ${NODES} nodes (${GPUS_TOTAL} GPUs) for ${N_POINTS} points"

# `--gpus-per-node` has no short form, and `-G` is NOT it: `-G/--gpus` is the
# TOTAL across the allocation. Shortened to `-G ${GPUS_PER_NODE}` this asks for
# 4 GPUs spread over ${NODES} nodes, which sbatch rejects outright once NODES > 4
# ("Failed to validate job spec, --gpus < -N") and, for NODES <= 4, silently
# grants one GPU per node instead of four -- the sweep still finishes, in four
# times the waves. Leave it long.
JOB=$(sbatch --parsable \
  -qregular -Cgpu -Am3246_g -t02:00:00 \
  -N "${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  -Jcubic_sweep \
  -o "${SWEEP_DIR}/slurm-%j.log" \
  --export="ALL,SWEEP_DIR=${SWEEP_DIR},N_POINTS=${N_POINTS},GPUS_TOTAL=${GPUS_TOTAL},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

step="srun --exact -n1 -N1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for i in $(seq 0 $((N_POINTS - 1))); do
  $step bash -c "
      uv run ran sweep ran \
          --s-index='${i}' --sweep-dir='${SWEEP_DIR}' --n-points='${N_POINTS}'
    " > "${SWEEP_DIR}/point_$(printf '%02d' "${i}").log" 2>&1 &

  while (( $(jobs -rp | wc -l) >= GPUS_TOTAL )); do
    wait -n || true
  done
  sleep 0.5
done
wait || true

uv run ran sweep collect --sweep-dir="${SWEEP_DIR}" --n-points="${N_POINTS}"
EOF
)

echo "Submitted packed job: ${JOB}"
echo "Logs:    ${SWEEP_DIR}/slurm-${JOB}.log  (+ per-point point_NN.log)"
echo "Results: ${SWEEP_DIR}/results.npz and ${SWEEP_DIR}/wasserstein_vs_s.pdf"
