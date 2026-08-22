#!/bin/bash
# Launch the cubic-response RAN-vs-IBU sweep as ONE packed multi-node job.
# Run on the login node:  bash scripts/submit_sweep.sh
# (Do NOT sbatch this file itself; it computes the sweep dir and submits the job.)
#
# Why packed instead of an array of 25 single-node jobs: on Perlmutter the queue
# wait for a 2-hour 1-node job can be ~1 day, while 4-7 node jobs slide into a
# ~2-hour wait window. We grab a few nodes at once (4 A100 GPUs per node) and run
# the sweep points concurrently, one GPU per point, via a bash background loop +
# srun step placement. collect runs inline at the end of the same job.
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
N_POINTS=24

# NODES=6 -> 24 GPUs == 24 points -> every point runs at once, one GPU each
# (single wave, no risk of a 2nd wave overrunning the wall clock). Lower NODES to
# use fewer GPUs at the cost of extra waves; all sit in the same Perlmutter queue
# window. Keep GPUS_TOTAL >= N_POINTS for the clean one-point-per-GPU mapping.
NODES=${NODES:-6}
GPUS_PER_NODE=4
GPUS_TOTAL=$((NODES * GPUS_PER_NODE))

SWEEP_DIR="${PROJECT_DIR}/runs/cubic_sweep_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${SWEEP_DIR}"
echo "Sweep dir: ${SWEEP_DIR}"
echo "Requesting ${NODES} nodes (${GPUS_TOTAL} GPUs) for ${N_POINTS} points"

# Submit a single batch job. The heredoc is the batch script; it is quoted
# ('EOF') so $i / $SWEEP_DIR / $(...) are evaluated at job runtime, not now.
# SWEEP_DIR, N_POINTS, GPUS_TOTAL reach the job via --export. --output lives on
# the command line because SLURM does not expand shell vars in #SBATCH lines.
JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --account=m3246_g --time=02:00:00 \
  --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  --job-name=cubic_sweep \
  --output="${SWEEP_DIR}/slurm-%j.log" \
  --export="ALL,SWEEP_DIR=${SWEEP_DIR},N_POINTS=${N_POINTS},GPUS_TOTAL=${GPUS_TOTAL},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

# Each point is one srun step pinned to a single GPU. --exact lets multiple steps
# share the allocation simultaneously (without it the first step grabs whole
# nodes and the rest block). --gpus-per-task=1 sets CUDA_VISIBLE_DEVICES per step
# so JAX in each point sees exactly one GPU -- which matters, because JAX
# preallocates most of the memory on every device it can see. ~16 cores / 56G per A100.
step="srun --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for i in $(seq 0 $((N_POINTS - 1))); do
  # Capture the log so a crash in one point is easy to find and
  # never sinks the others (collect tolerates missing point files). Each point
  # runs both methods, so a point file is complete or absent, never half.
  $step bash -c "
      uv run ran sweep ran \
          --s-index='${i}' --sweep-dir='${SWEEP_DIR}' --n-points='${N_POINTS}'
    " > "${SWEEP_DIR}/point_$(printf '%02d' "${i}").log" 2>&1 &

  # Throttle to GPUS_TOTAL concurrent steps; start the next as soon as one frees
  # a GPU (dynamic, so an uneven N_POINTS-over-GPUS_TOTAL split wastes no idle
  # GPU time when NODES is lowered below the single-wave setting).
  # `|| true`: a point that crashes must not abort the sweep under set -e — it
  # just leaves its point file missing, and collect tolerates the gap.
  while (( $(jobs -rp | wc -l) >= GPUS_TOTAL )); do
    wait -n || true
  done
  sleep 0.5  # small stagger to avoid "srun: Job step creation temporarily disabled"
done
wait || true  # let the final wave of points finish (ignore individual failures)

# Join the per-point files into results.npz + wasserstein_vs_s.pdf (runs
# on the head node of the allocation; resilience to gaps is built into collect).
uv run ran sweep collect --sweep-dir="${SWEEP_DIR}" --n-points="${N_POINTS}"
EOF
)

echo "Submitted packed job: ${JOB}"
echo "Logs:    ${SWEEP_DIR}/slurm-${JOB}.log  (+ per-point point_NN.log)"
echo "Results: ${SWEEP_DIR}/results.npz and ${SWEEP_DIR}/wasserstein_vs_s.pdf"
