"""Contains the implementations for the experimental design methods

Implements the supported experimental design methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .emukit_wrapper import EmukitExperimentalDesign
from .skactiveml_wrapper import SkActiveMLExperimentalDesign

TASK = "experimental_design"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	emukit=EmukitExperimentalDesign, skactiveml=SkActiveMLExperimentalDesign
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for experimental design.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary
			of calibration implementations for experimental design.
	"""
	return IMPLEMENTATIONS


class ExperimentalDesignMethodModel(CalibrationModel):
	"""The experimental design method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	query_strategy: str = Field(
		description="The active learning query strategy",
		default="greedy_sampling_target",
	)


class ExperimentalDesignMethod(CalibrationMethodBase):
	"""The experimental design method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ExperimentalDesignMethodModel,
		engine: str = "emukit",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""ExperimentalDesignMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ExperimentalDesignMethodModel): The calibration
				specification.
		    engine (str, optional): The experimental design backend.
				Defaults to "emukit".
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
