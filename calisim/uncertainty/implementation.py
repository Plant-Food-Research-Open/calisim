"""Contains the implementations for the uncertainty analysis methods

Implements the supported uncertainty analysis methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .chaospy_wrapper import ChaospyUncertaintyAnalysis
from .openturns_wrapper import OpenTurnsUncertaintyAnalysis
from .pygpc_wrapper import PygpcUncertaintyAnalysis

TASK = "uncertainty_analysis"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	chaospy=ChaospyUncertaintyAnalysis,
	pygpc=PygpcUncertaintyAnalysis,
	openturns=OpenTurnsUncertaintyAnalysis,
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for the uncertainty analysis.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for the uncertainty analysis.
	"""
	return IMPLEMENTATIONS


class UncertaintyAnalysisMethodModel(CalibrationModel):
	"""The uncertainty analysis method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	flatten_Y: bool = Field(description="Flatten the simulation outputs", default=False)
	order: int = Field(
		description="The order for polynomial chaos expansion", default=2
	)
	solver: str | list[str] = Field(
		description="The solver for performing the uncertainty analysis",
		default="linear",
	)
	algorithm: str = Field(
		description="The algorithm for the uncertainty analysis", default=""
	)


class UncertaintyAnalysisMethod(CalibrationMethodBase):
	"""The uncertainty analysis method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: UncertaintyAnalysisMethodModel,
		engine: str = "chaospy",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""UncertaintyAnalysisMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (UncertaintyAnalysisMethodModel): The calibration
				specification.
		    engine (str, optional): The uncertainty analysis backend.
				Defaults to "chaospy".
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
