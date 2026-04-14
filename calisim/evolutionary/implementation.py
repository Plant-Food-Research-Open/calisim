"""Contains the implementations for the evolutionary algorithm methods

Implements the supported evolutionary algorithm methods.

"""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "evolutionary"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	spotpy=f"calisim.{TASK}.spotpy_wrapper:SPOTPYEvolutionary",
	evotorch=f"calisim.{TASK}.evotorch_wrapper:EvoTorchEvolutionary",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for evolutionary algorithm.

	Returns:
		Dict[str, str]: The dictionary
			of calibration implementations for evolutionary algorithm.
	"""
	return BASE_IMPLEMENTATIONS


class EvolutionaryMethodModel(CalibrationModel):
	"""The evolutionary algorithm method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	objective: str | None = Field(description="The objective function", default="rmse")
	directions: list[str] | None = Field(
		description="The list of objective directions", default=["minimize"]
	)
	operators: dict[str, Any] | None = Field(
		description="The dictionary of evolutionary operators", default=None
	)


class EvolutionaryMethod(CalibrationMethodBase):
	"""The evolutionary algorithm method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: EvolutionaryMethodModel,
		engine: str = "spotpy",
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""EvolutionaryMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (EvolutionaryMethodModel): The calibration
				specification.
		    engine (str, optional): The evolutionary algorithm backend.
				Defaults to "spotpy".
			implementation (type[CalibrationWorkflowBase] | None): The
				calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			get_implementations(),
			implementation,
		)
