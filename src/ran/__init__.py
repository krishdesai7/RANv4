"""RAN: Reweighting Adversarial Networks.

Importing anything under ``ran`` first pins the Keras 3 backend to JAX and
enables JAX's 64-bit mode. Both settings are read once, when ``jax``/``keras``
are first imported, so they have to be in place before any submodule pulls
those in -- putting them here means every entry point gets them for free.

RAN is float64 end to end (see :mod:`ran.models`), which JAX silently downcasts
to float32 unless x64 mode is on. To trade that precision for GPU throughput,
set ``JAX_ENABLE_X64=0`` in the environment and switch the ``dtype=`` arguments
in :mod:`ran.models` to ``"float32"``.

``setdefault`` throughout, so an explicit environment override still wins --
that is how :mod:`ran.baselines.omnifold` pins itself back to TensorFlow.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Only relevant to ran.baselines.omnifold, the one module that still runs on
# TensorFlow; harmless everywhere else.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
