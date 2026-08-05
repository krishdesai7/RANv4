# Baselines

This directory contains the baselines for the RAN project.

## IBU

IBU (Iterative Bayesian Unfolding) baseline to compare with RAN. It is a simple unfolding method that uses a Bayesian approach to unfold the data.

It is implemented in the [**`ibu.py`**](ibu.py) file.
Usage:

```zsh
uv run -m ran baseline ibu --run-dir runs/2026-...
uv run -m ran baseline ibu --run-dir runs # all runs
```

IBU performs 1D per-variable unfolding with purity-based automatic binning. It builds the response matrix from MC, unfolds data, and converts the result to per-event weights for evaluation with the same metrics as RAN and <span style="font-variant: small-caps;">OmniFold</span>.
