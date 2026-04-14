"""Contains the implementations for the Approximate Bayesian Computation methods

Implements the supported Approximate Bayesian Computation methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "abc"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	pymc=f"calisim.{TASK}.pymc_wrapper:PyMCApproximateBayesianComputation",
	pyabc=f"calisim.{TASK}.pyabc_wrapper:PyABCApproximateBayesianComputation",
	elfi=f"calisim.experimental.{TASK}.elfi_wrapper:ELFIApproximateBayesianComputation",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for Approximate Bayesian Computation.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for Approximate Bayesian Computation.
	"""
	return BASE_IMPLEMENTATIONS


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
	quantile: float = Field(
		description="Selection quantile used for the sample acceptance threshold",
		default=0.2,
	)


class ApproximateBayesianComputationMethod(CalibrationMethodBase):
	"""The Approximate Bayesian Computation method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ApproximateBayesianComputationMethodModel,
		engine: str = "pymc",
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""ApproximateBayesianComputationMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ApproximateBayesianComputationMethodModel): The
				calibration specification.
		    engine (str, optional): The Approximate Bayesian
				Computation backend. Defaults to "pymc".
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
