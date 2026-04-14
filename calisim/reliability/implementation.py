"""Contains the implementations for the reliability analysis methods

Implements the supported reliability analysis methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "reliability"
BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	openturns=f"calisim.{TASK}.openturns_wrapper:OpenTurnsReliabilityAnalysis",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for reliability analysis.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for reliability analysis.
	"""
	return BASE_IMPLEMENTATIONS


class ReliabilityAnalysisMethodModel(CalibrationModel):
	"""The reliability analysis method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration
			base model class.
	"""

	comparison: str = Field(
		description="The comparison for the probability of a threshold event occurring",
		default="less_or_equal",
	)
	threshold: float = Field(
		description="The threshold for comparing against an event occurring", default=0
	)


class ReliabilityAnalysisMethod(CalibrationMethodBase):
	"""The reliability analysis method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ReliabilityAnalysisMethodModel,
		engine: str = "openturns",
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""ReliabilityAnalysisMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ReliabilityAnalysisMethodModel): The calibration
				specification.
		    engine (str, optional): The reliability analysis backend.
				Defaults to "openturns".
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
