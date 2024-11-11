"""Contains the implementations for the Bayesian quadrature methods

Implements the supported Bayesian quadrature methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .emukit_wrapper import EmukitBayesianQuadrature

TASK = "bayesian_quadrature"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	emukit=EmukitBayesianQuadrature
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for Bayesian quadrature.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary
			of calibration implementations for Bayesian quadrature.
	"""
	return IMPLEMENTATIONS


class BayesianQuadratureMethodModel(CalibrationModel):
	"""The Bayesian quadrature method data model.

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


class BayesianQuadratureMethod(CalibrationMethodBase):
	"""The Bayesian quadrature method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: BayesianQuadratureMethodModel,
		engine: str = "emukit",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""HistoryMatchingMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (BayesianQuadratureMethodModel): The calibration
				specification.
		    engine (str, optional): The Bayesian quadrature backend.
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
