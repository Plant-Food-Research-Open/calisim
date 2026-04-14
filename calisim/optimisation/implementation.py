"""Contains the implementations for the optimisation methods

Implements the supported optimisation methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "optimisation"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	optuna=f"calisim.{TASK}.optuna_wrapper:OptunaOptimisation",
	emukit=f"calisim.{TASK}.emukit_wrapper:EmukitOptimisation",
	openturns=f"calisim.{TASK}.openturns_wrapper:OpenTurnsOptimisation",
	botorch=f"calisim.{TASK}.botorch_wrapper:BoTorchOptimisation",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for optimisation.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]:
			The dictionary of calibration implementations for optimisation.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


class OptimisationMethodModel(CalibrationModel):
	"""The optimisation method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	directions: list[str] | None = Field(
		description="The list of objective directions", default=["minimize"]
	)
	acquisition_func: str | None = Field(
		description="The acquisition function for Bayesian optimisation", default="ei"
	)
	use_saasbo: bool = Field(
		description="Enable Sparse Axis-Aligned Subspace Bayesian Optimization",
		default=False,
	)


class OptimisationMethod(CalibrationMethodBase):
	"""The optimisation method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: OptimisationMethodModel,
		engine: str = "optuna",
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""OptimisationMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (OptimisationMethodModel): The calibration
				specification.
		    engine (str, optional): The optimisation backend.
				Defaults to "optuna".
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
