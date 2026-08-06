#!/bin/bash
set -uo pipefail
cd "${PROJECT_DIR}"

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
  echo "seed=${SEED}"
  echo "data_seed=${DATA_SEED}"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${BENCH_DIR}/provenance.txt"

# `import ran` FIRST: it is what pins KERAS_BACKEND, and importing keras on its
# own reports the stock default instead -- which is how the smoke run came to
# record "backend tensorflow" for the JAX arm. jax.devices() is the check that
# actually matters: a silent CPU fallback would void the whole comparison,
# and JAX skips the CUDA backend quietly when it sees no GPU.
uv run python -c "
import sys
import ran  # noqa: F401  -- pins KERAS_BACKEND before keras loads
import jax, keras
print('keras', keras.__version__, 'backend', keras.backend.backend())
print('jax', jax.__version__, 'default backend', jax.default_backend())
print('jax devices', jax.devices())
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
# Wall clock and peak RSS per stage, appended as one JSON object per line.
# GNU time carries the RSS; without it we still get wall clock from the shell.
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

bench_stage train \
  uv run -m ran train \
  --config="${CONFIG}" \
  --n-samples="${N_SAMPLES}" \
  --seed="${SEED}" \
  --data-seed="${DATA_SEED}" \
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

# Baselines are independent of each other; a failure in one must not cost us the
# other, nor the replot. Hence the trailing `|| true`.
bench_stage omnifold uv run -m ran baseline omnifold --run-dir="${RUN_DIR}" || true
bench_stage ibu      uv run -m ran baseline ibu      --run-dir="${RUN_DIR}" || true

# Re-enter with --load-run: reloads the trained generator, picks up the baseline
# weight files written above, and regenerates the plots with all three overlays.
bench_stage replot uv run -m ran train --load-run="${RUN_DIR}" || true

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
