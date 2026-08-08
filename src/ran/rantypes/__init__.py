"""Records, constants and aliases shared across the package.

These live apart from the code that uses them for one concrete reason: a
process gets a single Keras backend, fixed at the first `keras` import, so
`ran.cli` and `ran.baselines._shared` have to stay importable without
committing to one. They cannot reach into `ran.train` (JAX) or
`ran.baselines.omnifold` (TensorFlow) for a shared declaration, so the
declaration lives here instead. Nothing in this package imports keras or jax
at runtime.

Types owned by exactly one module stay with that module: `TrainResult` and
`TrainState` are in `ran.train`.
"""

from .configs import REQUIRED_KEYS as REQUIRED_KEYS
from .configs import GaussianConfig as GaussianConfig
from .configs import RunConfig as RunConfig
from .constants import CACHE_DIR as CACHE_DIR
from .constants import CACHE_FILENAMES as CACHE_FILENAMES
from .constants import DEFAULT_PURITY_THRESHOLD as DEFAULT_PURITY_THRESHOLD
from .constants import GENERATORS as GENERATORS
from .constants import JET_OBS as JET_OBS
from .constants import N_FILES as N_FILES
from .constants import SUBSTRUCTURE_VARIABLES as SUBSTRUCTURE_VARIABLES
from .constants import ZENODO_RECORD as ZENODO_RECORD
from .constants import JetVarInfo as JetVarInfo
from .enums import DatasetName as DatasetName
from .enums import LogLevel as LogLevel
from .events import ZXY as ZXY
from .events import DatasetSplits as DatasetSplits
from .events import Events as Events
from .events import Populations as Populations
from .events import Split as Split
from .results import IBUResult as IBUResult
from .results import UnfoldingPopulations as UnfoldingPopulations
from .results import VariableOutcome as VariableOutcome
from .types import Batch as Batch
from .types import KerasVariable as KerasVariable
from .types import MetricRecord as MetricRecord
from .types import Nested as Nested
from .types import RANModel as RANModel
from .types import StatelessOptimizer as StatelessOptimizer
from .types import Variables as Variables
from .types import VarInfo as VarInfo
