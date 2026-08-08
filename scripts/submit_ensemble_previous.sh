#!/bin/bash
# Ensemble on the pre-migration arm (branch: master, TensorFlow).
#
#   Run on the login node:  SHARED=1 scripts/submit_ensemble_previous.sh
#
# The mirror of scripts/submit_ensemble.sh, and the other half of a two-sample
# comparison: without a spread for THIS arm too, a difference between the arms
# cannot be told from master having drawn one lucky initialization.
#
# master's main() takes no --seed and its train() takes none either, so its
# weight init comes from system entropy afresh in every process. Members are
# therefore just N repeats -- independent draws by construction, at the cost of
# none of them being individually replayable. Its data seed is hardcoded to 42
# inside RAN_Dataset, so the events are held fixed across members for free,
# which is exactly the pinning the jax arm has to ask for.
#
# Members are ~9m16s here against ~45s on the jax arm, almost all of it the
# per-process tf.data shuffle over the full sample. Cost-optimal allocation
# under a 13x cost ratio puts ~3.6 jax members on every master member, so the
# default is 5 here against 8 (or more) there. Raise N_MEMBERS only after
# seeing whether the SD from 5 is already small next to the gap being tested.
#
# Knobs (environment):
#   N_MEMBERS=5   repeats; ~9.5 min each, so budget TIME accordingly
#   SHARED=1, LOCAL=1, CONFIG, N_SAMPLES, GPUS, TIME, QOS
#                 as in scripts/submit_previous.sh
# Anything passed on the command line is forwarded to every training call.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/global/u1/k/kdesai/RANv4}
CONFIG=${CONFIG:-params/2d_correlated.yaml}
ARM=tf
N_MEMBERS=${N_MEMBERS:-5}
GPUS=${GPUS:-1}
N_SAMPLES=${N_SAMPLES:-500000}
# 5 x ~9m16s is ~47 minutes; the rest is slack. Over-request rather than lose
# the last member to the wall clock -- there is no resume.
TIME=${TIME:-02:30:00}

if [[ ${SHARED:-0} == 1 ]]; then
  QOS=${QOS:-shared}
  NODE_FLAGS=(--gpus="${GPUS}")
else
  QOS=${QOS:-regular}
  NODE_FLAGS=(--nodes=1 --gpus-per-node="${GPUS}")
fi

BENCH_DIR="${PROJECT_DIR}/bench/ensemble_${ARM}_init_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${BENCH_DIR}"
printf '%s\n' "$@" > "${BENCH_DIR}/train_args"

echo "Arm:        ${ARM} (TensorFlow), train only"
echo "Bench dir:  ${BENCH_DIR}"
echo "Config:     ${CONFIG}   n_samples=${N_SAMPLES}"
echo "Members:    ${N_MEMBERS}   (unseeded init; data seed hardcoded to 42)"
echo "Resources:  ${GPUS} GPU, qos=${QOS}, time=${TIME}"

# master's ran/__init__.py is empty, so nothing pins the backend; its train.py
# is written against raw tf ops. Pin it here rather than trust ~/.keras.
export KERAS_BACKEND=tensorflow

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
  echo "vary=init"
  echo "n_members=${N_MEMBERS}"
  echo "seed=unseeded"
  echo "data_seed=42"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${BENCH_DIR}/provenance.txt"

uv run python -c "
import sys
import keras
print('keras', keras.__version__, 'backend', keras.backend.backend())
print('python', sys.version.split()[0])
import tensorflow as tf
print('tensorflow', tf.__version__)
print('tf GPUs', tf.config.list_physical_devices('GPU'))
" > "${BENCH_DIR}/env.txt" 2>&1 || true

HAVE_GNU_TIME=0
/usr/bin/time -v -o /dev/null true > /dev/null 2>&1 && HAVE_GNU_TIME=1

EXTRA=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && EXTRA+=("${line}")
done < "${BENCH_DIR}/train_args"

mkdir -p runs
FAILED=0

for ((i = 0; i < N_MEMBERS; i++)); do
  log="${BENCH_DIR}/train_member${i}.txt"
  tv="${BENCH_DIR}/train_member${i}.time"

  before=$(mktemp); after=$(mktemp)
  ls -1d runs/*/ 2> /dev/null | sort > "${before}"

  echo "=== [member ${i}] $(date -u +%H:%M:%S) ==="
  start=$(date +%s)
  # No --seed: master has none. Each process draws its own init from entropy,
  # which is what makes these independent samples.
  if [[ ${HAVE_GNU_TIME} == 1 ]]; then
    /usr/bin/time -v -o "${tv}" \
      uv run -m ran \
      --config="${CONFIG}" \
      --n_samples="${N_SAMPLES}" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1
    rc=$?
  else
    uv run -m ran \
      --config="${CONFIG}" \
      --n_samples="${N_SAMPLES}" \
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

  printf '{"stage":"train_member%s","seed":null,"data_seed":42,"wall_s":%d,"max_rss_kb":%s,"exit":%d,"run_dir":"%s"}\n' \
    "${i}" "${wall}" "${rss}" "${rc}" "${run_dir}" >> "${BENCH}"
  echo "=== [member ${i}] exit=${rc} wall=${wall}s run_dir=${run_dir} ==="

  if [[ ${rc} -ne 0 || -z "${run_dir}" ]]; then
    echo "WARNING: member ${i} failed; see ${log}"
    FAILED=$((FAILED + 1))
    continue
  fi
  cp "${run_dir}/metrics.json" "${BENCH_DIR}/metrics_member${i}.json" 2> /dev/null \
    || echo "WARNING: member ${i} wrote no metrics.json"
done

sacct -j "${SLURM_JOB_ID:-0}" \
  --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqTRES%40,State \
  > "${BENCH_DIR}/sacct.txt" 2>&1 || true

echo
echo "===================== SUMMARY (ensemble, ${ARM}) ====================="
cat "${BENCH}"
echo "failed members: ${FAILED}"
echo
# scripts/ensemble_spread.py is stdlib-only and lives on both branches, so it
# runs here on master unchanged.
uv run scripts/ensemble_spread.py "${BENCH_DIR}" || true
EOF
chmod +x "${BENCH_DIR}/job.sh"

export PROJECT_DIR BENCH_DIR CONFIG N_SAMPLES ARM N_MEMBERS KERAS_BACKEND

if [[ ${LOCAL:-0} == 1 ]]; then
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "LOCAL=1 outside a Slurm allocation: this would run on a login node," >&2
    echo "with no GPU. Get one first, e.g." >&2
    echo "  salloc -A m3246_g -C gpu -q interactive -t 01:00:00 -N 1 --gpus 1" >&2
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
  --job-name="ran_ens_tf" \
  --output="${BENCH_DIR}/slurm-%j.log" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},BENCH_DIR=${BENCH_DIR},CONFIG=${CONFIG},N_SAMPLES=${N_SAMPLES},N_MEMBERS=${N_MEMBERS},ARM=${ARM},KERAS_BACKEND=tensorflow" \
  "${BENCH_DIR}/job.sh")

echo
echo "Submitted: ${JOB}"
echo "Watch:     tail -f ${BENCH_DIR}/slurm-${JOB}.log"
echo "Spread:    uv run scripts/ensemble_spread.py ${BENCH_DIR}"
