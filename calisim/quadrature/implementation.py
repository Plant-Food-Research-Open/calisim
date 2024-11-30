"""Contains the implementations for the quadrature methods

Implements the supported quadrature methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .emukit_wrapper import EmukitQuadrature

TASK = "quadrature"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	emukit=EmukitQuadrature
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for quadrature.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary
			of calibration implementations for quadrature.
	"""
	return IMPLEMENTATIONS


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
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""QuadratureMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (QuadratureMethodModel): The calibration
				specification.
		    engine (str, optional): The Quadrature backend.
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
