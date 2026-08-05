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

## <span style="font-variant: small-caps;">OmniFold</span>

<span style="font-variant: small-caps;">OmniFold</span> baseline to compare with RAN. It is a deep learning-based unfolding method that uses a Bayesian approach to unfold the data.

It is implemented in the [**`omnifold.py`**](omnifold.py) file.
Usage:

```zsh
uv run -m ran baseline omnifold --run-dir runs/2026-...
uv run -m ran baseline omnifold --run-dir runs # all runs
```

RAN itself runs on the JAX backend, but the third-party `omnifold` package does not: its `weighted_binary_crossentropy` calls raw `tf.gather` on the label tensor, which raises `TracerArrayConversionError` the moment JAX traces it. So this module pins the backend back to TensorFlow.

A process gets one Keras backend, set at first `keras` import, so invoke the <span style="font-variant: small-caps;">OmniFold</span> baseline in its own process with `uv run -m ran baseline omnifold`; never import it from a module that has already touched JAX. The cubic sweep keeps the two sides in separate subcommands for exactly this reason.

### `ran.baselines.omnifold::omnifold_unfold`

Trains <span style="font-variant: small-caps;">OmniFold</span> on in-memory arrays and return mean-normalized gen weights.

Trains on (data reco = `x_data`, MC reco = `x_sim`, MC gen = `z_gen`), then reweights `z_target` (defaults to `z_gen`) through the gen-level model. Returns a 1D weight array, normalized so its mean is 1.
