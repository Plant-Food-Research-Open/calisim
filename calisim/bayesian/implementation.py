"""Contains the implementations for the Bayesian calibration methods

Implements the supported Bayesian calibration methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "bayesian"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	openturns=f"calisim.{TASK}.openturns_wrapper:OpenTurnsBayesianCalibration",
	emcee=f"calisim.{TASK}.emcee_wrapper:EmceeBayesianCalibration",
	dynesty=f"calisim.{TASK}.dynesty_wrapper:DynestyBayesianCalibration",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for Bayesian calibration.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for Bayesian calibration.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


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
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""BayesianCalibrationMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (BayesianCalibrationMethodModel): The
				calibration specification.
		    engine (str, optional): The Bayesian calibration
				backend. Defaults to "openturns".
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
