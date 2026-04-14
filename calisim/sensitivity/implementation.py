"""Contains the implementations for the sensitivity analysis methods

Implements the supported sensitivity analysis methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "sensitivity"
BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	salib=f"calisim.{TASK}.salib_wrapper:SALibSensitivityAnalysis",
	openturns=f"calisim.{TASK}.openturns_wrapper:OpenTurnsSensitivityAnalysis",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for sensitivity analysis.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for sensitivity analysis.
	"""
	return BASE_IMPLEMENTATIONS


class SensitivityAnalysisMethodModel(CalibrationModel):
	"""The sensitivity analysis method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration
			base model class.
	"""

	order: int = Field(
		description="The order for the polynomial chaos expansion", default=2
	)


class SensitivityAnalysisMethod(CalibrationMethodBase):
	"""The sensitivity analysis method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: SensitivityAnalysisMethodModel,
		engine: str = "salib",
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""SensitivityAnalysisMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (SensitivityAnalysisMethodModel): The calibration
				specification.
		    engine (str, optional): The sensitivity analysis backend.
				Defaults to "salib".
			implementation (type[CalibrationWorkflowBase] | None): The calibration
				workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			get_implementations(),
			implementation,
		)
