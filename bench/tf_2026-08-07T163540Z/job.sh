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

# KERAS_BACKEND is exported above, so keras picks it up here too. Recorded so
# the arm's backend is evidence in the bench dir rather than an assumption.
uv run python -c "
import sys
import keras, tensorflow as tf
print('keras', keras.__version__, 'backend', keras.backend.backend())
print('tensorflow', tf.__version__)
print('tf GPUs', tf.config.list_physical_devices('GPU'))
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
  # .txt, not .log: .gitignore excludes *.log, and these per-stage logs are
  # the first thing anyone wants when a stage misbehaves.
  local log="${BENCH_DIR}/${name}.txt"
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
  echo "FATAL: training failed (rc=${TRAIN_RC}) or produced no run dir; see ${BENCH_DIR}/train.txt"
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
