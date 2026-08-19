import os

os.environ.setdefault(key="KERAS_BACKEND", value="jax")
os.environ.setdefault(key="JAX_ENABLE_X64", value="1")
# .baselines.omnifold is the sole module that runs TensorFlow
os.environ.setdefault(key="TF_CPP_MIN_LOG_LEVEL", value="2")

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
