# Benchmarks

## `boundary.py`

How is the wall clock time distributed between host numpy and the training loop?

```zsh
uv run benchmarks/boundary.py
```

The primary metric is the ratio of host numpy time to training loop time. Training scales with the accelerator; the scipy metrics and the npz cache write do not. On CPU numpy looks cheap. On an A100 the same numpy may be most of the run, which is what decides whether porting `_wd_per_dim`/`_js_per_dim` to jax.numpy is worthwhile.

## `precision.py`

Does float32 cost unfolding accuracy, or is the difference just seed variance?

One run proves nothing: RAN is an adversarial min-max game, so two runs at different seeds in the _same_ dtype already differ by ~1 percentage point per dimension. Run an ensemble in each dtype and compare the two distributions. The script runs two ensembles in each dtype, and compares them.

```zsh
for s in $(seq 0 9); do
    print "Running seed $s"
    uv run benchmarks/precision.py float64 "$s" | tee -a f64.log
    uv run benchmarks/precision.py float32 "$s" | tee -a f32.log
done
uv run benchmarks/compare_precision.py f64.log f32.log 0.5
```

Each run prints one SUMMARY line, which is what the comparison reads.

## `compare_precision.py`

Compare two precision ensembles produced by `precision.py`.

```zsh
uv run benchmarks/compare_precision.py f64.log f32.log [margin_pp]
```

The runs are **paired**: the same `--seed` initializes the same weights in both arms, on the same data. Pairing is most of the available statistical power here, because seed-to-seed variation (sd ~0.4pp) is several times the effect being looked for (~0.2pp), and it cancels in the within-pair difference.

The following are common pitfalls in significance testing of ensembles of paired data:

### 1. Comparing the separation of means to the _pooled standard deviation_

Runs should be compared be the spread of their means, not the spread of individual runs. The right scale is the standard error of the difference of means, smaller by $\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$.

### 2. Treating paired runs as independent samples

Paired runs that share a common initial condition are not independent samples. Treating them as independent samples will inflate the statistical power of the test.

### 3. Computing "$n$ needed for power $p$" from the _observed_ effect

An effect estimated near $p=0.05$ at small $n$ is inflated, so that figure is far too small. A t-test can only ever fail to find a difference, so the affirmative question -- "is the gap small enough?" -- is answered by TOST against a stated margin.

The parameter `margin_pp` is the margin of equivalence, in percentage points, that sets an upper bound on the effect size that is considered to be negligible. It is a physics judgement and has to be stated, so it is an argument, rather than an computed result.
