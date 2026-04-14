"""Contains the implementations for the quadrature methods

Implements the supported quadrature methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "quadrature"
BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	emukit=f"calisim.{TASK}.emukit_wrapper:EmukitQuadrature",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for quadrature.

	Returns:
		Dict[str, str]: The dictionary
			of calibration implementations for quadrature.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


class QuadratureMethodModel(CalibrationModel):
	"""The quadrature method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	kernel: str | None = Field(
		description="The Kernel embeddings for Bayesian quadrature",
		default="QuadratureRBFLebesgueMeasure",
	)
	measure: str | None = Field(
		description="The Integration measures", default="LebesgueMeasure"
	)


class QuadratureMethod(CalibrationMethodBase):
	"""The quadrature method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: QuadratureMethodModel,
		engine: str = "emukit",
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""QuadratureMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (QuadratureMethodModel): The calibration
				specification.
		    engine (str, optional): The Quadrature backend.
				Defaults to "emukit".
			implementation (type[CalibrationWorkflowBase] | CalibrationWorkflowBase
				| None): The calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			get_implementations(),
			implementation,
		)
