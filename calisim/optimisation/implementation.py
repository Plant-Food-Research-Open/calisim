"""Contains the implementations for the optimisation methods

Implements the supported optimisation methods.

"""

from collections.abc import Callable

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import IntervalCalibrationModel
from .botorch_wrapper import BoTorchOptimisation
from .optuna_wrapper import OptunaOptimisation

TASK = "optimisation"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	optuna=OptunaOptimisation, botorch=BoTorchOptimisation
)


def get_optimisation_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for optimisation.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]:
			The dictionary of calibration implementations for optimisation.
	"""
	return IMPLEMENTATIONS


class OptimisationMethodModel(IntervalCalibrationModel):
	"""The optimisation method data model.

	Args:
	    BaseModel (IntervalCalibrationModel):
	        The calibration base model class.
	"""

	directions: list[str] | None = ["minimize"]


class OptimisationMethod(CalibrationMethodBase):
	"""The optimisation method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: OptimisationMethodModel,
		engine: str = "optuna",
	) -> None:
		"""OptimisationMethod constructor.

		Args:
			calibration_func (Callable):
				The calibration function.
				For example, a simulation function or objective function.
		    specification (OptimisationMethodModel):
		        The calibration specification.
		    engine (str, optional):
		        The optimisation backend. Defaults to "optuna".
		"""
		super().__init__(calibration_func, specification, TASK, engine, IMPLEMENTATIONS)
