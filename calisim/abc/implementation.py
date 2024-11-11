"""Contains the implementations for the Approximate Bayesian Computation methods

Implements the supported Approximate Bayesian Computation methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .pyabc_wrapper import PyABCApproximateBayesianComputation
from .pymc_wrapper import PyMCApproximateBayesianComputation

TASK = "approximate_bayesian_computation"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	pymc=PyMCApproximateBayesianComputation, pyabc=PyABCApproximateBayesianComputation
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for Approximate Bayesian Computation.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for Approximate Bayesian Computation.
	"""
	return IMPLEMENTATIONS


class ApproximateBayesianComputationMethodModel(CalibrationModel):
	"""The Approximate Bayesian Computation method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	n_bootstrap: int = Field(description="The number of bootstrap samples", default=5)
	min_population_size: int = Field(
		description="The minimum population size", default=5
	)
	epsilon: float = Field(
		description="The dissimilarity threshold between observed and simulated data",
		default=0,
	)
	sum_stat: str | Callable = Field(
		description="The summary statistic function for observed and simulated data",
		default="identity",
	)
	distance: str | Callable | None = Field(
		description="The distance function between observed and simulated data",
		default=None,
	)


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
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ApproximateBayesianComputationMethodModel): The
				calibration specification.
		    engine (str, optional): The Approximate Bayesian
				Computation backend. Defaults to "pymc".
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
