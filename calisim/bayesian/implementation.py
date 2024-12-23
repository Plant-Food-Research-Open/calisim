"""Contains the implementations for the Bayesian calibration methods

Implements the supported Bayesian calibration methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .emcee_wrapper import EmceeBayesianCalibration
from .openturns_wrapper import OpenTurnsBayesianCalibration

TASK = "bayesian_calibration"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	openturns=OpenTurnsBayesianCalibration, emcee=EmceeBayesianCalibration
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for Bayesian calibration.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for Bayesian calibration.
	"""
	return IMPLEMENTATIONS


class BayesianCalibrationMethodModel(CalibrationModel):
	"""The Bayesian calibration method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	log_density: bool = Field(
		description="Take the log of the target density.", default=False
	)
	initial_state: list | bool = Field(
		description="Initial state of the chain.", default=None
	)
	moves: dict[str, float] = Field(
		description="List of methods for updating coordinates of ensemble walkers.",
		default=None,
	)


class BayesianCalibrationMethod(CalibrationMethodBase):
	"""The Bayesian calibration method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: BayesianCalibrationMethodModel,
		engine: str = "openturns",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""BayesianCalibrationMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (BayesianCalibrationMethodModel): The
				calibration specification.
		    engine (str, optional): The Bayesian calibration
				backend. Defaults to "openturns".
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
