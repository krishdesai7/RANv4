# RAN: Reweighting Adversarial Networks

`ran` is a library for training and evaluating reweighting adversarial networks (RANs).

Importing anything under `ran` first pins the Keras 3 backend to JAX and enables JAX's 64-bit mode. Both settings are read once, when `jax`/`keras` are first imported, so they must be in place before any submodule imports either.

`ran` defaults to float64 end to end (see :mod:`ran.models`), which JAX silently downcasts to float32 unless x64 mode is on. To trade that precision for GPU throughput,
set `JAX_ENABLE_X64=0` in the environment and switch the `dtype=` arguments in :mod:`ran.models` to `"float32"`.

`setdefault` throughout, so that the environment can be explicitly overridden. That is how :mod:`ran.baselines.omnifold` pins itself back to TensorFlow.

Importing this package must not import `keras`. :mod:`ran.baselines.omnifold` hard-sets the backend to TensorFlow at import, and :mod:`ran.train` refuses to load on anything but JAX, so a package `__init__` that pulled in both would make the two mutually unimportable, and would leak `KERAS_BACKEND=tensorflow` into every subprocess besides.

Import the submodule needed (`from ran.workflow import run`); the CLI re-exports below are the sole exception, and they defer their own imports into the command bodies.
