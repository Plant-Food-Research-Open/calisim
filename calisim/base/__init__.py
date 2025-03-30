import importlib

from .calibration_base import CalibrationMethodBase, CalibrationWorkflowBase
from .emukit_base import EmukitBase
from .example_model_base import ExampleModelBase
from .history_matching_base import HistoryMatchingBase
from .openturns_base import OpenTurnsBase
from .surrogate_base import SurrogateBase

__all__ = [
	CalibrationMethodBase,
	CalibrationWorkflowBase,
	EmukitBase,
	ExampleModelBase,
	HistoryMatchingBase,
	OpenTurnsBase,
	SurrogateBase,
]

if importlib.util.find_spec("torch") is not None:
	from .sbi_base import SimulationBasedInferenceBase

	__all__.append(SimulationBasedInferenceBase)

if importlib.util.find_spec("elfi") is not None:
	from .elfi_base import ELFIBase

	__all__.append(ELFIBase)
