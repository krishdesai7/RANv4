#!/bin/bash
# Run 72 more paired float64/float32 precision-benchmark seeds and append the
# results to f64.log / f32.log at the repo root, same files the manual loop
# in benchmarks/precision.py's docstring writes to.
# Run on the login node:  bash scripts/submit_precision.sh
# (Do NOT sbatch this file itself; it computes the seed list and submits the job.)
#
# f64.log/f32.log already cover seeds 0-9 plus 12,23,32,58,96,117,132,153,166,170
# (20 paired seeds). SEED_START/SEED_END below pick a disjoint block so there's
# no need to cross-check for collisions.
#
# One node = 4 A100x40G on Perlmutter. 72 seeds x 2 dtypes = 144 independent
# runs, so we throttle them across the node's 4 GPUs with srun --exact steps in
# a background loop -- the same pattern scripts/submit_sweep.sh already uses
# successfully on this account/queue. Each prior paired run took ~48s
# (see commit 692ccfb..35279c1), so 144 runs / 4 GPUs should finish in under
# 30 min; --time below is 1.5x that (45 min) -- comfortable margin while still
# landing in a favorable queue window.
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
SEED_START=200
SEED_END=271  # inclusive -> 72 seeds

NODES=1
GPUS_PER_NODE=4

RUN_DIR="${PROJECT_DIR}/runs/precision_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${RUN_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Seeds ${SEED_START}-${SEED_END} (72 paired seeds, 144 runs) on ${GPUS_PER_NODE} GPUs"

# Submit a single batch job. The heredoc is the batch script; it is quoted
# ('EOF') so shell expansion happens at job runtime, not now. Everything the
# job needs reaches it via --export. --output lives on the command line
# because SLURM does not expand shell vars in #SBATCH lines.
JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --account=m3246_g --time=00:45:00 \
  --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  --job-name=precision_bench \
  --output="${RUN_DIR}/slurm-%j.log" \
  --export="ALL,RUN_DIR=${RUN_DIR},SEED_START=${SEED_START},SEED_END=${SEED_END},GPUS_PER_NODE=${GPUS_PER_NODE},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

# Each run is one srun step pinned to a single GPU. --exact lets multiple
# steps share the allocation simultaneously (without it the first step grabs
# the whole node and the rest block). --gpus-per-task=1 sets
# CUDA_VISIBLE_DEVICES per step so JAX in each run sees exactly one GPU --
# which matters, because JAX preallocates most of the memory on every device
# it can see. ~16 cores / 56G per A100 (4 GPUs, 64 cores, 256G on a Perlmutter
# GPU node).
step="srun --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  for dtype in float64 float32; do
    # Capture each run's own log so a crash in one run is easy to find and
    # never sinks the others or corrupts the aggregate file.
    $step bash -c "
        uv run python benchmarks/precision.py '${dtype}' '${seed}'
      " > "${RUN_DIR}/${dtype}_seed$(printf '%04d' "${seed}").log" 2>&1 &

    # Throttle to GPUS_PER_NODE concurrent steps; start the next as soon as
    # one frees a GPU. `|| true`: a run that crashes must not abort the batch
    # under set -e -- it just leaves no SUMMARY line in its log, and the
    # collection step below tolerates the gap.
    while (( $(jobs -rp | wc -l) >= GPUS_PER_NODE )); do
      wait -n || true
    done
    sleep 0.5  # small stagger to avoid "srun: Job step creation temporarily disabled"
  done
done
wait || true  # let the final wave finish (ignore individual failures)

# Pull the SUMMARY line out of each per-run log, in seed order, and append to
# the same f64.log/f32.log the manual loop writes to. A run that crashed
# leaves no SUMMARY line and is silently skipped here; check its own log file
# under RUN_DIR to see why.
for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  grep -h '^SUMMARY' "${RUN_DIR}/float64_seed$(printf '%04d' "${seed}").log" >> "${PROJECT_DIR}/f64.log" || true
  grep -h '^SUMMARY' "${RUN_DIR}/float32_seed$(printf '%04d' "${seed}").log" >> "${PROJECT_DIR}/f32.log" || true
done

n64=$(grep -c '^SUMMARY' "${PROJECT_DIR}/f64.log" || true)
n32=$(grep -c '^SUMMARY' "${PROJECT_DIR}/f32.log" || true)
echo "f64.log now has ${n64} SUMMARY lines; f32.log now has ${n32}"
EOF
)

echo "Submitted job: ${JOB}"
echo "Logs:    ${RUN_DIR}/slurm-${JOB}.log  (+ per-run <dtype>_seedNNNN.log)"
echo "Results: appended to ${PROJECT_DIR}/f64.log and ${PROJECT_DIR}/f32.log"
