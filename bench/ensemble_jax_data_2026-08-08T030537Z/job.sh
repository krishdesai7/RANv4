#!/bin/bash
set -uo pipefail
cd "${PROJECT_DIR}"

BENCH="${BENCH_DIR}/bench.jsonl"
: > "${BENCH}"

{
  echo "arm=${ARM}"
  echo "mode=ensemble"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "config=${CONFIG}"
  echo "n_samples=${N_SAMPLES}"
  echo "vary=${VARY}"
  echo "seeds=${SEEDS}"
  echo "seed=${SEED}"
  echo "data_seed=${DATA_SEED}"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${BENCH_DIR}/provenance.txt"

# See scripts/submit.sh: `import ran` must come first, and jax.devices() is the
# check that catches a silent CPU fallback.
uv run python -c "
import sys
import ran  # noqa: F401  -- pins KERAS_BACKEND before keras loads
import jax, keras
print('keras', keras.__version__, 'backend', keras.backend.backend())
print('jax', jax.__version__, 'default backend', jax.default_backend())
print('jax devices', jax.devices())
print('python', sys.version.split()[0])
" > "${BENCH_DIR}/env.txt" 2>&1 || true

HAVE_GNU_TIME=0
/usr/bin/time -v -o /dev/null true > /dev/null 2>&1 && HAVE_GNU_TIME=1

EXTRA=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && EXTRA+=("${line}")
done < "${BENCH_DIR}/train_args"

mkdir -p runs
FAILED=0

for s in ${SEEDS}; do
  # Whichever axis VARY names takes the sweep value; the other is pinned.
  if [[ ${VARY} == data ]]; then
    m_seed=${SEED}
    m_data_seed=${s}
  else
    m_seed=${s}
    m_data_seed=${DATA_SEED}
  fi

  log="${BENCH_DIR}/train_member${s}.txt"
  tv="${BENCH_DIR}/train_member${s}.time"

  before=$(mktemp); after=$(mktemp)
  ls -1d runs/*/ 2> /dev/null | sort > "${before}"

  echo "=== [member ${s}: seed=${m_seed} data_seed=${m_data_seed}] $(date -u +%H:%M:%S) ==="
  start=$(date +%s)
  if [[ ${HAVE_GNU_TIME} == 1 ]]; then
    /usr/bin/time -v -o "${tv}" \
      uv run -m ran train \
      --config="${CONFIG}" \
      --n-samples="${N_SAMPLES}" \
      --seed="${m_seed}" \
      --data-seed="${m_data_seed}" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1
    rc=$?
  else
    uv run -m ran train \
      --config="${CONFIG}" \
      --n-samples="${N_SAMPLES}" \
      --seed="${m_seed}" \
      --data-seed="${m_data_seed}" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1
    rc=$?
  fi
  wall=$(($(date +%s) - start))

  rss=null
  if [[ -f "${tv}" ]]; then
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${tv}")
    [[ -z "${rss}" ]] && rss=null
  fi

  # Ask the run itself which directory it wrote. The runs/ diff below cannot
  # tell our new directory from one a concurrent job created in the same
  # window, and would silently attribute another job's metrics to this member.
  # Both arms log this line verbatim.
  run_dir=$(sed -n 's|.*Saved run to \(runs/[^ ]*\).*|\1|p' "${log}" | head -1)
  if [[ -z "${run_dir}" ]]; then
    ls -1d runs/*/ 2> /dev/null | sort > "${after}"
    run_dir=$(comm -13 "${before}" "${after}" | tail -1)
  fi
  rm -f "${before}" "${after}"
  run_dir=${run_dir%/}

  printf '{"stage":"train_member%s","seed":%s,"data_seed":%s,"wall_s":%d,"max_rss_kb":%s,"exit":%d,"run_dir":"%s"}\n' \
    "${s}" "${m_seed}" "${m_data_seed}" "${wall}" "${rss}" "${rc}" "${run_dir}" >> "${BENCH}"
  echo "=== [member ${s}] exit=${rc} wall=${wall}s run_dir=${run_dir} ==="

  # One bad member must not cost the others; the aggregator reports how many it
  # actually found rather than assuming all of them landed.
  if [[ ${rc} -ne 0 || -z "${run_dir}" ]]; then
    echo "WARNING: member ${s} failed; see ${log}"
    FAILED=$((FAILED + 1))
    continue
  fi
  cp "${run_dir}/metrics.json" "${BENCH_DIR}/metrics_member${s}.json" 2> /dev/null \
    || echo "WARNING: member ${s} wrote no metrics.json"
done

sacct -j "${SLURM_JOB_ID:-0}" \
  --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqTRES%40,State \
  > "${BENCH_DIR}/sacct.txt" 2>&1 || true

echo
echo "===================== SUMMARY (ensemble) ====================="
cat "${BENCH}"
echo "failed members: ${FAILED}"
echo
uv run scripts/ensemble_spread.py "${BENCH_DIR}" \
  ${REFERENCE:+--reference="${REFERENCE}"} || true
