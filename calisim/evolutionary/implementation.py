"""Contains the implementations for the evolutionary algorithm methods

Implements the supported evolutionary algorithm methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .spotpy_wrapper import SPOTPYEvolutionary

TASK = "evolutionary"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	spotpy=SPOTPYEvolutionary
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for evolutionary algorithm.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary
			of calibration implementations for evolutionary algorithm.
	"""
	return IMPLEMENTATIONS


class EvolutionaryMethodModel(CalibrationModel):
	"""The evolutionary algorithm method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	objective: str | None = Field(description="The objective function", default="rmse")


class EvolutionaryMethod(CalibrationMethodBase):
	"""The evolutionary algorithm method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: EvolutionaryMethodModel,
		engine: str = "spotpy",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""EvolutionaryMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (EvolutionaryMethodModel): The calibration
				specification.
		    engine (str, optional): The evolutionary algorithm backend.
				Defaults to "spotpy".
			implementation (CalibrationWorkflowBase | None): The
				calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			IMPLEMENTATIONS,
			implementation,
		)
