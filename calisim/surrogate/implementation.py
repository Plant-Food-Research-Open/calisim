"""Contains the implementations for the surrogate modelling methods

Implements the supported surrogate modelling methods.

"""

import importlib
from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .sklearn_wrapper import SklearnSurrogateModel

TASK = "surrogate_modelling"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	sklearn=SklearnSurrogateModel
)

if importlib.util.find_spec("gpytorch") is not None:
	from .gpytorch_wrapper import GPyTorchSurrogateModel

	IMPLEMENTATIONS["gpytorch"] = GPyTorchSurrogateModel


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for surrogate modelling.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for surrogate modelling.
	"""
	return IMPLEMENTATIONS


class SurrogateModelMethodModel(CalibrationModel):
	"""The surrogate modelling method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	batch_size: int = Field(
		description="The batch size when training the surrogate model", default=1000
	)
	flatten_Y: bool = Field(description="Flatten the simulation outputs", default=False)


class SurrogateModelMethod(CalibrationMethodBase):
	"""The surrogate modelling method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: SurrogateModelMethodModel,
		engine: str = "sklearn",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""SurrogateModelMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (SurrogateModelMethodModel): The calibration
				specification.
		    engine (str, optional): The surrogate modelling backend.
				Defaults to "sklearn".
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
