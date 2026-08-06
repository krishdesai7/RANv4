#!/bin/bash
# Full RAN run on Perlmutter for the pre-migration arm (branch: master, TensorFlow).
#
#   Run on the login node:  scripts/submit_previous.sh
#   (Do NOT sbatch this file itself; it computes the bench dir and submits.)
#
# The mirror image of scripts/submit.sh on refactor/jax: same four stages, same
# instrumentation, same bench.jsonl schema, so the two are directly comparable.
# Only the CLI differs -- master predates the typer rewrite and drives everything
# through python-fire, one entry point per module:
#
#   jax arm                                 master arm
#   ------------------------------------    ------------------------------------
#   uv run -m ran train --n-samples=N       uv run -m ran --n_samples=N
#   uv run -m ran baseline omnifold ...     uv run -m ran.baselines.omnifold ...
#   uv run -m ran baseline ibu ...          uv run -m ran.baselines.ibu ...
#   uv run -m ran train --load-run=D        uv run -m ran --load_run=D
#
# Two further differences worth knowing when reading the numbers:
#   - master's main() takes no --seed, so its weight init is unseeded and its
#     run is not replayable. Data seed is hardcoded to 42, which is what the jax
#     arm passes explicitly, so both arms do see identical events and batches.
#   - master's ran/__init__.py is empty: nothing pins KERAS_BACKEND, so the
#     backend falls out of ~/.keras/keras.json. Its train.py is written against
#     raw tf ops and @tf.function, so we pin TensorFlow here rather than trust
#     whatever that file happens to say.
#
# Knobs (environment):
#   SMOKE=1     tiny run -- validates the whole pipeline in minutes before
#               committing to the real thing. Do this first.
#   LOCAL=1     run here instead of submitting, for use inside an existing
#               salloc. The way to smoke-test when a queue is refusing jobs:
#                 salloc -A m3246_g -C gpu -q interactive -t 00:30:00 -N 1 --gpus 1
#                 SMOKE=1 LOCAL=1 scripts/submit_previous.sh
#               Refuses to run outside an allocation (that would mean a login
#               node, with no GPU).
#   CONFIG=...  Gaussian config (default params/2d_correlated.yaml)
#   GPUS, TIME, QOS, N_SAMPLES
# Anything passed on the command line is forwarded to the training entry point.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/global/u1/k/kdesai/RANv4}
CONFIG=${CONFIG:-params/2d_correlated.yaml}
ARM=tf

# Neither implementation distributes across devices, so one GPU is what a run
# can actually use. More would idle.
GPUS=${GPUS:-1}

if [[ ${SMOKE:-0} == 1 ]]; then
  QOS=${QOS:-debug}
  TIME=${TIME:-00:20:00}
  N_SAMPLES=${N_SAMPLES:-20000}
  TAG=smoke_${ARM}
else
  QOS=${QOS:-regular}
  # Longer than the jax arm's default: this is the slower implementation and the
  # one whose wall time we do not yet know. Better to over-request than to lose
  # a run to the wall clock and have nothing to compare.
  TIME=${TIME:-06:00:00}
  N_SAMPLES=${N_SAMPLES:-500000}
  TAG=${ARM}
fi

BENCH_DIR="${PROJECT_DIR}/bench/${TAG}_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${BENCH_DIR}"

printf '%s\n' "$@" > "${BENCH_DIR}/train_args"

echo "Arm:        ${ARM} (TensorFlow, pre-migration)"
echo "Bench dir:  ${BENCH_DIR}"
echo "Config:     ${CONFIG}   n_samples=${N_SAMPLES}"
echo "Resources:  ${GPUS} GPU, qos=${QOS}, time=${TIME}"

cat > "${BENCH_DIR}/job.sh" <<'EOF'
#!/bin/bash
set -uo pipefail
cd "${PROJECT_DIR}"

# master's package __init__ is empty, so the backend would otherwise come from
# ~/.keras/keras.json -- which the jax arm may well have rewritten.
export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL=2

BENCH="${BENCH_DIR}/bench.jsonl"
: > "${BENCH}"

# ---------------------------------------------------------------- provenance
{
  echo "arm=${ARM}"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "config=${CONFIG}"
  echo "n_samples=${N_SAMPLES}"
  echo "seed=unseeded"
  echo "data_seed=42"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${BENCH_DIR}/provenance.txt"

uv run python -c "
import keras, sys
print('keras', keras.__version__, 'backend', keras.backend.backend())
print('python', sys.version.split()[0])
" > "${BENCH_DIR}/env.txt" 2>&1 || true
nvidia-smi >> "${BENCH_DIR}/env.txt" 2>&1 || true

# ------------------------------------------------------------- GPU sampling
GPU_PID=""
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 5 > "${BENCH_DIR}/gpu.csv" 2>/dev/null &
  GPU_PID=$!
fi
cleanup() { [[ -n "${GPU_PID}" ]] && kill "${GPU_PID}" 2> /dev/null; }
trap cleanup EXIT

# ------------------------------------------------------------------- staging
# Probe for GNU time rather than just testing for the file: BSD time is also at
# /usr/bin/time and supports neither -v nor -o, so mere existence proves nothing.
HAVE_GNU_TIME=0
/usr/bin/time -v -o /dev/null true > /dev/null 2>&1 && HAVE_GNU_TIME=1

bench_stage() {
  local name="$1"; shift
  local log="${BENCH_DIR}/${name}.log"
  local tv="${BENCH_DIR}/${name}.time"
  local start end wall rss rc

  echo "=== [${name}] $(date -u +%H:%M:%S) ==="
  start=$(date +%s)
  if [[ ${HAVE_GNU_TIME} == 1 ]]; then
    /usr/bin/time -v -o "${tv}" "$@" > "${log}" 2>&1
    rc=$?
  else
    "$@" > "${log}" 2>&1
    rc=$?
  fi
  end=$(date +%s)
  wall=$((end - start))

  rss=null
  if [[ -f "${tv}" ]]; then
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${tv}")
    [[ -z "${rss}" ]] && rss=null
  fi

  printf '{"stage":"%s","wall_s":%d,"max_rss_kb":%s,"exit":%d}\n' \
    "${name}" "${wall}" "${rss}" "${rc}" >> "${BENCH}"
  echo "=== [${name}] exit=${rc} wall=${wall}s rss=${rss}kB ==="
  return ${rc}
}

# Identify the run directory training creates by diffing runs/ around it --
# more robust than `ls -t | head -1`, which would pick up the other arm's runs.
before=$(mktemp); after=$(mktemp)
mkdir -p runs
ls -1d runs/*/ 2> /dev/null | sort > "${before}"

# Read one arg per line. A `while read` loop rather than `mapfile` so this
# stays runnable under bash 3.2 for local testing; printf wrote a single empty
# line when there were no args, and the -n test drops it. The ${EXTRA[@]+...}
# guard below keeps an empty array from tripping `set -u` on bash before 4.4.
EXTRA=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && EXTRA+=("${line}")
done < "${BENCH_DIR}/train_args"

# fire takes underscored flags, and there is no `train` subcommand: the module
# entry point *is* main().
bench_stage train \
  uv run -m ran \
  --config="${CONFIG}" \
  --n_samples="${N_SAMPLES}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
TRAIN_RC=$?

ls -1d runs/*/ 2> /dev/null | sort > "${after}"
RUN_DIR=$(comm -13 "${before}" "${after}" | tail -1)
rm -f "${before}" "${after}"

if [[ ${TRAIN_RC} -ne 0 || -z "${RUN_DIR}" ]]; then
  echo "FATAL: training failed (rc=${TRAIN_RC}) or produced no run dir; see ${BENCH_DIR}/train.log"
  exit 1
fi
RUN_DIR=${RUN_DIR%/}
echo "run_dir=${RUN_DIR}" >> "${BENCH_DIR}/provenance.txt"
echo "Run dir: ${RUN_DIR}"

bench_stage omnifold uv run -m ran.baselines.omnifold --run_dir="${RUN_DIR}" || true
bench_stage ibu      uv run -m ran.baselines.ibu      --run_dir="${RUN_DIR}" || true

# Re-enter with --load_run so the plots pick up the baseline overlays.
bench_stage replot uv run -m ran --load_run="${RUN_DIR}" || true

# --------------------------------------------------------------- accounting
cleanup; GPU_PID=""
sleep 2
sacct -j "${SLURM_JOB_ID}" \
  --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqTRES%40,State \
  > "${BENCH_DIR}/sacct.txt" 2>&1 || true

cp -r "${RUN_DIR}" "${BENCH_DIR}/run_artifacts" 2> /dev/null || true

echo
echo "===================== SUMMARY (${ARM}) ====================="
cat "${BENCH}"
echo "Run dir:   ${RUN_DIR}"
echo "Bench dir: ${BENCH_DIR}"
EOF
chmod +x "${BENCH_DIR}/job.sh"

# The job body reads these from the environment either way: sbatch forwards them
# with --export, an in-allocation run inherits them from here.
export PROJECT_DIR BENCH_DIR CONFIG N_SAMPLES ARM

if [[ ${LOCAL:-0} == 1 ]]; then
  # Run right here instead of submitting -- for use inside an salloc, which is
  # the way to smoke-test when a queue is refusing jobs. Also the only way to
  # watch the harness itself work.
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "LOCAL=1 outside a Slurm allocation: this would run on a login node," >&2
    echo "with no GPU. Get one first, e.g." >&2
    echo "  salloc -A m3246_g -C gpu -q interactive -t 00:30:00 -N 1 --gpus 1" >&2
    echo "Set FORCE_LOGIN_NODE=1 to override (you almost certainly do not want to)." >&2
    [[ ${FORCE_LOGIN_NODE:-0} == 1 ]] || exit 2
  fi
  echo "Running in allocation ${SLURM_JOB_ID:-<none>} (no sbatch)"
  echo
  bash "${BENCH_DIR}/job.sh" 2>&1 | tee "${BENCH_DIR}/run.log"
  rc=${PIPESTATUS[0]}
  echo
  echo "Finished rc=${rc}"
  echo "Timings:   ${BENCH_DIR}/bench.jsonl"
  exit "${rc}"
fi

JOB=$(sbatch --parsable \
  --qos="${QOS}" --constraint=gpu --account=m3246_g --time="${TIME}" \
  --nodes=1 --gpus-per-node="${GPUS}" \
  --job-name="ran_${TAG}" \
  --output="${BENCH_DIR}/slurm-%j.log" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},BENCH_DIR=${BENCH_DIR},CONFIG=${CONFIG},N_SAMPLES=${N_SAMPLES},ARM=${ARM}" \
  "${BENCH_DIR}/job.sh")

echo
echo "Submitted: ${JOB}"
echo "Watch:     tail -f ${BENCH_DIR}/slurm-${JOB}.log"
echo "Timings:   ${BENCH_DIR}/bench.jsonl"
