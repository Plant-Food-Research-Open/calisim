"""Contains the implementations for the uncertainty analysis methods

Implements the supported uncertainty analysis methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "uncertainty"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	chaospy=f"calisim.{TASK}.chaospy_wrapper:ChaospyUncertaintyAnalysis",
	pygpc=f"calisim.{TASK}.pygpc_wrapper:PygpcUncertaintyAnalysis",
	openturns=f"calisim.{TASK}.openturns_wrapper:OpenTurnsUncertaintyAnalysis",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for the uncertainty analysis.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for the uncertainty analysis.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


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
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""UncertaintyAnalysisMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (UncertaintyAnalysisMethodModel): The calibration
				specification.
		    engine (str, optional): The uncertainty analysis backend.
				Defaults to "chaospy".
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
