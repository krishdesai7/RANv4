#!/bin/bash
# HEP ensemble on the JAX arm: N trainings that differ only in weight init.
#
#   Run on the login node:  SHARED=1 scripts/submit_ensemble.sh
#
# `data_seed` is held fixed so every member sees identical events, splits and
# batch order; only `--seed` moves. The spread across members is therefore the
# model uncertainty from initialization alone, which is the number needed to say
# whether a single run's result is a real effect or run-to-run noise.
#
# Trains only. The baselines are init-independent -- OmniFold and IBU would
# produce the same answer every member and cost ~7 minutes each doing it -- and
# `ran train` already writes metrics.json, which is all the aggregator reads.
#
# Members run sequentially inside ONE job on ONE GPU, not in parallel across a
# node. A member is ~45s, so eight of them is ~6 minutes of compute -- less than
# the queue wait for an exclusive node, and a quarter of the node-hours. Two
# concrete things would also have to be fixed first: JAX preallocates the whole
# card (a second process on it OOMs unless XLA_PYTHON_CLIENT_MEM_FRACTION is
# set), and run directories are timestamped to the second, so members starting
# together would collide on a name and on the runs/ diff that finds it.
#
# Knobs (environment):
#   SEEDS="0 1 ... 7"   weight-init seeds, one member each
#   DATA_SEED=42        held fixed across members -- this is the whole point
#   SHARED=1, LOCAL=1, CONFIG, N_SAMPLES, GPUS, TIME, QOS
#                       as in scripts/submit.sh
# Anything passed on the command line is forwarded to every `ran train`.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/global/u1/k/kdesai/RANv4}
CONFIG=${CONFIG:-params/2d_correlated.yaml}
ARM=jax
# Eight members: the sample SD carries ~27% relative error at n=8 against ~35%
# at n=5, and at ~45s a member the extra three are close to free.
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
DATA_SEED=${DATA_SEED:-42}
GPUS=${GPUS:-1}
N_SAMPLES=${N_SAMPLES:-500000}
# ~6 minutes of work; the rest is slack for a slow node or a cold cache.
TIME=${TIME:-00:45:00}

if [[ ${SHARED:-0} == 1 ]]; then
  QOS=${QOS:-shared}
  NODE_FLAGS=(--gpus="${GPUS}")
else
  QOS=${QOS:-regular}
  NODE_FLAGS=(--nodes=1 --gpus-per-node="${GPUS}")
fi

BENCH_DIR="${PROJECT_DIR}/bench/ensemble_${ARM}_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${BENCH_DIR}"
printf '%s\n' "$@" > "${BENCH_DIR}/train_args"

echo "Arm:        ${ARM} (JAX), train only"
echo "Bench dir:  ${BENCH_DIR}"
echo "Config:     ${CONFIG}   n_samples=${N_SAMPLES}"
echo "Seeds:      ${SEEDS}   (data_seed=${DATA_SEED}, fixed)"
echo "Resources:  ${GPUS} GPU, qos=${QOS}, time=${TIME}"

cat > "${BENCH_DIR}/job.sh" <<'EOF'
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
  echo "seeds=${SEEDS}"
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
  log="${BENCH_DIR}/train_seed${s}.txt"
  tv="${BENCH_DIR}/train_seed${s}.time"

  before=$(mktemp); after=$(mktemp)
  ls -1d runs/*/ 2> /dev/null | sort > "${before}"

  echo "=== [seed ${s}] $(date -u +%H:%M:%S) ==="
  start=$(date +%s)
  if [[ ${HAVE_GNU_TIME} == 1 ]]; then
    /usr/bin/time -v -o "${tv}" \
      uv run -m ran train \
      --config="${CONFIG}" \
      --n-samples="${N_SAMPLES}" \
      --seed="${s}" \
      --data-seed="${DATA_SEED}" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1
    rc=$?
  else
    uv run -m ran train \
      --config="${CONFIG}" \
      --n-samples="${N_SAMPLES}" \
      --seed="${s}" \
      --data-seed="${DATA_SEED}" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1
    rc=$?
  fi
  wall=$(($(date +%s) - start))

  rss=null
  if [[ -f "${tv}" ]]; then
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${tv}")
    [[ -z "${rss}" ]] && rss=null
  fi

  ls -1d runs/*/ 2> /dev/null | sort > "${after}"
  run_dir=$(comm -13 "${before}" "${after}" | tail -1)
  rm -f "${before}" "${after}"
  run_dir=${run_dir%/}

  printf '{"stage":"train_seed%s","seed":%s,"wall_s":%d,"max_rss_kb":%s,"exit":%d,"run_dir":"%s"}\n' \
    "${s}" "${s}" "${wall}" "${rss}" "${rc}" "${run_dir}" >> "${BENCH}"
  echo "=== [seed ${s}] exit=${rc} wall=${wall}s run_dir=${run_dir} ==="

  # One bad member must not cost the other four; the aggregator reports how
  # many it actually found rather than assuming all of them landed.
  if [[ ${rc} -ne 0 || -z "${run_dir}" ]]; then
    echo "WARNING: seed ${s} failed; see ${log}"
    FAILED=$((FAILED + 1))
    continue
  fi
  cp "${run_dir}/metrics.json" "${BENCH_DIR}/metrics_seed${s}.json" 2> /dev/null \
    || echo "WARNING: seed ${s} wrote no metrics.json"
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
EOF
chmod +x "${BENCH_DIR}/job.sh"

export PROJECT_DIR BENCH_DIR CONFIG N_SAMPLES ARM SEEDS DATA_SEED
export REFERENCE="${REFERENCE:-}"

if [[ ${LOCAL:-0} == 1 ]]; then
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "LOCAL=1 outside a Slurm allocation: this would run on a login node," >&2
    echo "with no GPU. Get one first, e.g." >&2
    echo "  salloc -A m3246_g -C gpu -q interactive -t 00:45:00 -N 1 --gpus 1" >&2
    echo "Set FORCE_LOGIN_NODE=1 to override (you almost certainly do not want to)." >&2
    [[ ${FORCE_LOGIN_NODE:-0} == 1 ]] || exit 2
  fi
  echo "Running in allocation ${SLURM_JOB_ID:-<none>} (no sbatch)"
  echo
  bash "${BENCH_DIR}/job.sh" 2>&1 | tee "${BENCH_DIR}/run.txt"
  exit "${PIPESTATUS[0]}"
fi

JOB=$(sbatch --parsable \
  --qos="${QOS}" --constraint=gpu --account=m3246_g --time="${TIME}" \
  "${NODE_FLAGS[@]}" \
  --job-name="ran_ensemble" \
  --output="${BENCH_DIR}/slurm-%j.log" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},BENCH_DIR=${BENCH_DIR},CONFIG=${CONFIG},N_SAMPLES=${N_SAMPLES},SEEDS=${SEEDS},DATA_SEED=${DATA_SEED},ARM=${ARM},REFERENCE=${REFERENCE}" \
  "${BENCH_DIR}/job.sh")

echo
echo "Submitted: ${JOB}"
echo "Watch:     tail -f ${BENCH_DIR}/slurm-${JOB}.log"
echo "Spread:    uv run scripts/ensemble_spread.py ${BENCH_DIR}"
