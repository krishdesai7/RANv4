#!/bin/bash
#
# Packed launcher for the bootstrap x seed variance design.
#
# One cell is one `ran uncertainty run`, so cells are independent and pack onto
# whatever GPUs the allocation has. Sizing, written out rather than guessed at:
# a cell is one ordinary training run (~3 min at the defaults below) plus the
# jet cache read, call it 4 minutes; B*S cells over NODES*4 GPUs is
# ceil(B*S / (NODES*4)) waves. The default 8x8 on two nodes is 8 waves, ~35
# minutes, inside the hour requested. Raising NODES cuts waves and buys queue
# time; it does not make a cell faster.
#
# Defaults measure the decomposition. For the bin-to-bin covariance the
# bootstrap axis is what needs replicates, not the seed one -- S can stay at 2
# because the interaction is already measured to 49 df by the 8x8 design:
#
#   B=100 S=2 NODES=4 TIME=01:30:00 bash scripts/submit_uncertainty.sh
#
# This is the grid the published numbers use (src/ran/uncertainty/README.md):
# a B=50 run confirmed against it (every lag correlation within 0.01, every
# effective rank within 0.2) before B=100 superseded it as the smaller grid's
# individual off-diagonal entries moved by up to 0.39 -- too much to publish
# a single matrix entry from, even though the aggregate structure was already
# right.
#
# Warm the jet cache on a login node first; a cold cache pulls 3.3GB from
# Zenodo inside the job.

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4

B=${B:-8}
S=${S:-8}
N_EVAL=${N_EVAL:-100000}
N_BINS=${N_BINS:-20}
CELLS=$((B * S))

NODES=${NODES:-2}
# Bump with the grid: the default 8x8 on two nodes is 8 waves (~35 min), but
# B=100 S=2 on four is 13 waves and would hit a one-hour wall.
TIME=${TIME:-01:00:00}
GPUS_PER_NODE=4
GPUS_TOTAL=$((NODES * GPUS_PER_NODE))

# The design measures RAN as the paper ships it, so these are the paper's
# values and not a cheaper stand-in. A design run at other settings is a
# variance budget for a model nobody is publishing.
RUN_ARGS=${RUN_ARGS:--Djets -n1000000 -l3 -u128}

DESIGN_DIR="${PROJECT_DIR}/runs/unc_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${DESIGN_DIR}"
echo "Design dir: ${DESIGN_DIR}"
echo "Running ${B}x${S} = ${CELLS} cells on ${GPUS_TOTAL} GPUs"

# `--gpus-per-node` has no short form, and `-G` is NOT it: `-G/--gpus` is the
# TOTAL across the allocation. See the same note in scripts/submit_hparam.sh.
JOB=$(sbatch --parsable \
  -qregular -Cgpu -Am3246_g -t"${TIME}" \
  -N "${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  -Jran_uncertainty \
  -o "${DESIGN_DIR}/slurm-%j.log" \
  --export="ALL,DESIGN_DIR=${DESIGN_DIR},B=${B},S=${S},CELLS=${CELLS},N_EVAL=${N_EVAL},N_BINS=${N_BINS},GPUS_TOTAL=${GPUS_TOTAL},RUN_ARGS=${RUN_ARGS},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

step="srun --exact -n1 -N1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for cell in $(seq 0 $((CELLS - 1))); do
  log="${DESIGN_DIR}/cell_$(printf '%04d' "${cell}").log"
  $step bash -c "
      uv run ran uncertainty run ${RUN_ARGS} \
          --cell '${cell}' --design-dir '${DESIGN_DIR}' \
          -B '${B}' -S '${S}' --n-eval '${N_EVAL}'
    " > "${log}" 2>&1 &

  while (( $(jobs -rp | wc -l) >= GPUS_TOTAL )); do
    wait -n || true
  done
  sleep 0.5
done
wait || true

uv run ran uncertainty collect \
    --design-dir "${DESIGN_DIR}" -B "${B}" -S "${S}" --n-bins "${N_BINS}"
EOF
)

echo "Submitted packed job: ${JOB}"
echo "Logs:    ${DESIGN_DIR}/slurm-${JOB}.log  (+ per-cell cell_NNNN.log)"
echo "Collect: uv run ran uncertainty collect --design-dir ${DESIGN_DIR} -B ${B} -S ${S}"
