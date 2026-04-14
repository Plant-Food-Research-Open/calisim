"""Contains the implementations for the experimental design methods

Implements the supported experimental design methods.

"""

from collections.abc import Callable

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "experimental_design"
BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	emukit=f"calisim.{TASK}.emukit_wrapper:EmukitExperimentalDesign",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for experimental design.

	Returns:
		Dict[str, str]: The dictionary
			of calibration implementations for experimental design.
	"""
	return BASE_IMPLEMENTATIONS


class ExperimentalDesignMethodModel(CalibrationModel):
	"""The experimental design method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""


class ExperimentalDesignMethod(CalibrationMethodBase):
	"""The experimental design method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ExperimentalDesignMethodModel,
		engine: str = "emukit",
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""ExperimentalDesignMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ExperimentalDesignMethodModel): The calibration
				specification.
		    engine (str, optional): The experimental design backend.
				Defaults to "emukit".
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
