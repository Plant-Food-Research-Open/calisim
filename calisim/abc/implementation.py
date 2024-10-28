"""Contains the implementations for the Approximate Bayesian Computation methods

Implements the supported Approximate Bayesian Computation methods.

"""

from collections.abc import Callable

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .pyabc_wrapper import PyABCApproximateBayesianComputation
from .pymc_wrapper import PyMCApproximateBayesianComputation

TASK = "approximate_bayesian_computation"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	pymc=PyMCApproximateBayesianComputation, pyabc=PyABCApproximateBayesianComputation
)


def get_abc_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for Approximate Bayesian Computation.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]:
			The dictionary of calibration implementations
			for Approximate Bayesian Computation.
	"""
	return IMPLEMENTATIONS


class ApproximateBayesianComputationMethodModel(CalibrationModel):
	"""The Approximate Bayesian Computation method data model.

	Args:
	    BaseModel (CalibrationModel):
	        The calibration base model class.
	"""

	n_bootstrap: int = 5
	min_population_size: int = 2
	epsilon: float = 0
	sum_stat: str | Callable = "identity"
	distance: str | Callable | None = None


class ApproximateBayesianComputationMethod(CalibrationMethodBase):
	"""The Approximate Bayesian Computation method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ApproximateBayesianComputationMethodModel,
		engine: str = "pymc",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""ApproximateBayesianComputationMethod constructor.

		Args:
			calibration_func (Callable):
				The calibration function.
				For example, a simulation function or objective function.
		    specification (ApproximateBayesianComputationMethodModel):
		        The calibration specification.
		    engine (str, optional):
		        The Approximate Bayesian Computation backend. Defaults to "pymc".
			implementation (CalibrationWorkflowBase | None):
				The calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			IMPLEMENTATIONS,
			implementation,
		)
