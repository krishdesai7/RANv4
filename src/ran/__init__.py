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

from . import baselines as baselines
from . import data as data
from . import experiments as experiments
from . import rantypes as rantypes
from .cli import app as app
from .cli import configure as configure
from .cli import evaluate_command as evaluate_command
from .cli import ibu_command as ibu_command
from .cli import leakage_check_command as leakage_check_command
from .cli import omnifold_command as omnifold_command
from .cli import sweep_collect_command as sweep_collect_command
from .cli import sweep_omnifold_command as sweep_omnifold_command
from .cli import sweep_ran_command as sweep_ran_command
from .cli import train_command as train_command
from .evaluate import apply_to_runs as apply_to_runs
from .evaluate import evaluate_run as evaluate_run
from .evaluate import evaluate_runs as evaluate_runs
from .evaluate import render_metrics as render_metrics
from .leakage import run_leakage_check as run_leakage_check
from .logging_config import configure_logging as configure_logging
from .models import build_discriminator as build_discriminator
from .models import build_generator as build_generator
from .plotting import plot_detector_level as plot_detector_level
from .plotting import plot_losses as plot_losses
from .plotting import plot_particle_level as plot_particle_level
from .train import normalize_weights as normalize_weights
from .train import train as train
from .train import weighted_bce as weighted_bce
