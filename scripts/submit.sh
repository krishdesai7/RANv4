#!/bin/bash
# End-to-end unfolding run: train -> IBU baseline -> replot with the baseline
# overlaid -> metrics. Defaults to all six jet observables, which is the
# full-fledged case; every stage runs in one sequential job on one GPU.
#
#   sbatch scripts/submit.sh                       # all 6 jet observables
#   sbatch scripts/submit.sh --seed 7              # extra flags reach `ran train`
#   sbatch scripts/submit.sh --var m --var w       # a subset
#   sbatch scripts/submit.sh --dataset gaussian --config params/2d_correlated.yaml
#
# The last form still works because TRAIN_ARGS below is *prepended*: click keeps
# the last occurrence of a scalar option, so anything on the command line wins.
# `--var` is the exception --- it is `multiple=True` and accumulates --- which is
# why the six-variable default comes from `ran train` itself (an empty `--var`
# means SUBSTRUCTURE_VARIABLES) rather than being spelled out there.
# One GPU, not four, and therefore `shared` rather than `regular`. Nothing in
# `ran train` shards across devices --- it is a single jitted program on device
# 0 --- so three of the four would sit idle while JAX preallocated ~75% of each.
# `shared` lets the job take a quarter of a node and be charged for a quarter of
# a node, and small jobs backfill into gaps a whole-node request cannot reach.
# The CPU and memory requests are the matching quarter of a Perlmutter GPU node
# (4x A100, 64 physical cores, 256GB): oversubscribing either is what makes the
# scheduler fall back to allocating the whole thing.
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#
# 30 minutes, not 2 hours. Measured on an A100 (`benchmarks/boundary.py`, same
# 500k x 6D shape): 4.6s to compile, 0.034s per epoch. Scaled to the parameters
# below the training term is ~15s; the pipeline is minutes, dominated by npz
# loading and matplotlib, not by the GPU. This is a ~5x margin.
#
# It assumes the jet cache is already warm. A cold cache pulls 3.3GB from Zenodo
# (Pythia26 1.55GB + Herwig 1.75GB) inside the job and will not fit --- warm it
# on a login node first, see the note in CLAUDE.md.
#SBATCH --time=00:30:00
#SBATCH --account=m3246_g
#SBATCH --job-name=ran_e2e
#SBATCH --output=slurm-%j.log

# `set -e` matters more here than in a one-liner: this is four stages, and a
# failed train must not go on to run IBU against a run directory from last week.
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
cd "${PROJECT_DIR}"

# RAN_CACHE_DIR is inherited from the submitting shell (SLURM exports the
# environment by default). It is deliberately not set here: relocating the cache
# re-downloads 3.3GB of Zenodo jet data on the first run, which should be a
# considered choice rather than a side effect of running this script.
echo "RAN_CACHE_DIR = ${RAN_CACHE_DIR:-<unset: using ./.cache>}"

# Give RAN its best shot on the full six-observable dataset. Every one of these
# is a scalar option, so passing it again on the command line overrides it.
#
#   -n 1000000  The Zenodo release holds ~1.6M jets per generator, and
#               `load_jet_dataset` refuses n_samples > n_avail. 1M leaves
#               headroom for the two generators' counts not matching exactly.
#   -l 3 -u 128 Width over depth. Andreassen et al. use 3x100 ReLU on exactly
#               these six observables; 3x128 is a strict superset of that
#               (~34k params vs ~17k for 5x64) while staying shallow, which
#               matters in a min-max game where depth destabilises the balance
#               between g and d faster than width does.
#   -P 100      Effectively disables early stopping: `still_running` is
#               `(epoch < n_epochs) & (wait < patience)` and n_epochs is 100.
#               Free, because the best state is restored *always*, not only
#               when early stopping fires --- so a longer run can only find a
#               better one. 100 epochs at 1M is ~13.6k generator updates.
#
# -b stays at the default 1024, and --seed stays unset on purpose: `train`
# draws one from system entropy and records it in config.json, so the run is
# reproducible after the fact without pinning it in advance.
TRAIN_ARGS=(--dataset jets -n 1000000 -l 3 -u 128 -P 100)

# `_save_run` names the run directory for the UTC timestamp and returns it only
# to its Python caller, so a shell has to find it. Anchor on a marker file
# rather than "newest directory": `runs/` already holds older runs, and a
# mistake here would silently attach IBU to one of them.
mkdir -p runs
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT

uv run ran train "${TRAIN_ARGS[@]}" "$@"

RUN_DIR="$(find runs -mindepth 1 -maxdepth 1 -type d -newer "${marker}" | sort | tail -1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "No run directory appeared under runs/ --- training did not save." >&2
  exit 1
fi
echo "Run dir: ${RUN_DIR}"

# IBU unfolds the same populations for comparison. The reload pass after it is
# what puts the baseline into the figures: `workflow.run` reads ibu_weights.npz
# only if it exists when the plots are drawn, and on the training pass it does
# not exist yet.
uv run ran baseline ibu --run-dir "${RUN_DIR}"
uv run ran train --load-run "${RUN_DIR}"

# The reload pass leaves metrics.json alone (it only forces on a fresh train),
# so recompute explicitly --- this is the number the run exists to produce.
uv run ran evaluate --run-dir "${RUN_DIR}" --force

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"

# Evidence that the persistent compilation cache did its job. A populated
# directory here is what makes the *next* run skip ~4.6s of XLA; an empty one
# means RAN_CACHE_DIR points somewhere unwritable and JAX only warned about it.
jax_cache="${RAN_CACHE_DIR:-.cache}/jax"
echo "XLA cache: $(find "${jax_cache}" -type f 2>/dev/null | wc -l | tr -d ' ') entries in ${jax_cache}"
