"""Contains the implementations for the sensitivity analysis methods

Implements the supported sensitivity analysis methods.

"""

from collections.abc import Callable

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .salib_wrapper import SALibSensitivityAnalysis

TASK = "sensitivity_analysis"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	salib=SALibSensitivityAnalysis
)


def get_sensitivity_analysis_implementations() -> (
	dict[str, type[CalibrationWorkflowBase]]
):
	"""Get the calibration implementations for sensitivity analysis.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for sensitivity analysis.
	"""
	return IMPLEMENTATIONS


class SensitivityAnalysisMethodModel(CalibrationModel):
	"""The sensitivity analysis method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration
			base model class.
	"""


class SensitivityAnalysisMethod(CalibrationMethodBase):
	"""The sensitivity analysis method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: SensitivityAnalysisMethodModel,
		engine: str = "salib",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""SensitivityAnalysisMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (SensitivityAnalysisMethodModel): The calibration
				specification.
		    engine (str, optional): The sensitivity analysis backend.
				Defaults to "salib".
			implementation (CalibrationWorkflowBase | None): The calibration
				workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			IMPLEMENTATIONS,
			implementation,
		)
