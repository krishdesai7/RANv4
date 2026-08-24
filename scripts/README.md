# Slurm scripts

## submit.sh

End-to-end unfolding run: train -> IBU baseline -> replot with the baseline overlaid -> metrics. Defaults to all six jet observables, which is the full-fledged case; every stage runs in one sequential job on one GPU.

### Examples

```zsh
    sbatch scripts/submit.sh                       # all 6 jet observables
    sbatch scripts/submit.sh --seed 7              # extra flags reach `ran train`
    sbatch scripts/submit.sh --var m --var w       # a subset
    sbatch scripts/submit.sh --dataset gaussian --config params/2d_correlated.yaml
```

### Queue and resource allocation

Since the default config uses GPU, not four, and therefore the queue is `shared` rather than `regular`. Nothing in `ran train` shards across devices. It is a single jitted program on device 0, so three of the four GPUs of an individual node would sit idle while JAX preallocated ~75% of each. `shared` lets the job take a quarter of a node and be charged for a quarter of a node, and small jobs backfill into gaps a whole-node request cannot reach.

_`-c32` is mandatory:_ the gpu_shared queue requires exactly 32 logical cores per GPU (a quarter of the node's 128) and rejects anything else. There is deliberately no `--mem` line. The scheduler converts a memory request into an equivalent core count and enforces the larger of the two, so `--mem=<M>GB` was silently a request for a different number of cores and cause a job failure. The ceiling that goes with 32 cores is ~54GB. Omitting it lets memory come out proportional to the cores, which is both correct and not a number that has to be rederived when the node spec changes. ~54GB is far more than an individual run needs: 1M events x 6 observables is ~105MB on device (z, x and y across all three splits) and well under a gigabyte on host.

`-Cgpu` sets the device requested to an 40GB A100, of which Perlmutter has ~1200 nodes; `-Cgpu&hbm80g` may be chosen if more memory is needed. This requests the 80GB part but from a pool of ~200, so it will cost queue time.

`-t00:15:00` sets the runtime to 15 minutes. With the default parameters, a full run takes about ~5 minutes end to end, dominated by npz loading and matplotlib, not by the GPU. This is a ~3x margin. This does however assume the jet cache is already warm. A cold cache pulls 3.3GB from Zenodo (Pythia26 1.55GB + Herwig 1.75GB), and the job may overrun the 15 minute limit. Warm it on a login node first with `uv run python -c "from ran.data import load_jet_dataset; load_jet_dataset(1000)"`.

### Run script

- `set -e` is important here because this script involves running four distinct stages, and a failed train must not go on to run IBU against a previous run directory.
- `RAN_CACHE_DIR` is inherited from the submitting shell (SLURM exports the environment by default).
- `_save_run` names the run directory for the UTC timestamp and returns it only to its Python caller, so a shell has to find it. Hence this script anchors on a marker file rather than "newest directory": `runs/` already holds older runs, and a mistake here would silently attach IBU to one of them.
- IBU unfolds the same populations for comparison. The reload pass after it puts the baseline into the figures: `workflow.run` reads ibu_weights.npz only if it exists when the plots are drawn, and on the training pass it does not exist yet.
- The reload pass does not update metrics.json (it only forces on a fresh train), so it must be recomputed explicitly.
- The script logs evidence that the persistent compilation cache did its job. A populated directory makes the _next_ run skip ~4.6s of XLA compilation; an empty one means `RAN_CACHE_DIR` points somewhere unwritable and JAX only warned about it.

### RAN Training Arguments

The default `TRAIN_ARGS` is _prepended_ to the command line arguments. Since click keeps the last occurrence of a scalar option, anything on the command line overrides the default. The only exception is `--var`, which is explicitly set to `multiple=True` and accumulates, which is why the six-variable default comes from `ran train` itself (an empty `--var` means `SUBSTRUCTURE_VARIABLES`) rather than being part of `TRAIN_ARGS`. The default `TRAIN_ARGS` is as follows:

- `-n1000000`: The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset` refuses $n_{\text{samples}} > n_{\text{avail}}$. 1M leaves headroom for the two generators' counts not matching exactly.
- `-l3 -u128`: Width over depth. Andreassen et al. use 3x100 ReLU on exactly these six observables; 3x128 follows that style staying shallow, which matters in a min-max game where depth destabilises the balance between g and d faster than width does.
- `-P100`: Effectively disables early stopping: `still_running` is `(epoch < n_epochs) & (wait < patience)` and `n_epochs` is 100. Free, because the best state is restored _always_, not only when early stopping fires, so a longer run can only find a better one. 100 epochs at 1M is ~13.6k generator updates.
- `-b1024`: Batch size.
- `--seed`: Seed for the random number generator. `train` draws one from system entropy and records it in config.json, so the run is reproducible after the fact without pinning it in advance.

## submit_precision.sh

- Runs 72 paired float64/float32 precision-benchmark seeds and appends the results to f64.log/f32.log at the repo root.
- Run it on a login node with `bash scripts/submit_precision.sh`. Do NOT sbatch this file itself; it computes the seed list and submits the job.
- One node = 4 A100x40G on Perlmutter. 72 seeds x 2 dtypes = 144 independent runs, so this script throttles them across the node's 4 GPUs with `srun --exact` steps in a background loop. 144 runs/4 GPUs should finish in under 30 min; `-t00:45:00` is 1.5x that; a comfortable margin while still landing in a favorable queue window.
- The heredoc is the batch script; it is quoted (`EOF`) so shell expansion happens at job runtime, not now. Everything the job needs reaches it via `--export`. `-o` lives on the command line because SLURM does not expand shell vars in `#SBATCH` lines.
- Each run is one `srun` step pinned to a single GPU. `--exact` lets multiple steps share the allocation simultaneously (without it the first step grabs the whole node and the rest block). `--gpus-per-task=1` sets `CUDA_VISIBLE_DEVICES` per step so JAX in each run sees exactly one GPU, which matters, because JAX preallocates most of the memory on every device it can see. ~16 cores/56GB per A100 (4 GPUs, 64 cores, 256GB on a Perlmutter GPU node).
- Each run's log is captured so a crash in one run is easy to find and does not affect the others or corrupt the aggregate file.
- The script throttles to `GPUS_PER_NODE` concurrent steps; starts the next as soon as one frees a GPU.
- `|| true` ensures a run that crashes does not abort the batch under `set -e`. It just leaves no `SUMMARY` line in its log, and the collection step tolerates the gap.
- Once the batch is complete, the script pulls the `SUMMARY` line out of each per-run log, in seed order, and appends to f64.log/f32.log. A run that crashed leaves no `SUMMARY` line and is silently skipped here; check its own log file under `RUN_DIR` to diagnose.

## submit_sweep.sh

This script is used to submit a job to the cluster. It is used to run the cubic-response RAN-vs-IBU sweep.

```

```
