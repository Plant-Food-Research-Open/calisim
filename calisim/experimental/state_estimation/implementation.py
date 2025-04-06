"""Contains the implementations for the state estimation methods

Implements the supported state estimation methods.

"""

from collections.abc import Callable

from pydantic import Field

from ...base import CalibrationMethodBase, CalibrationWorkflowBase
from ...data_model import CalibrationModel
from .ensemble_kalman_filter import EKFStateEstimation

TASK = "state_estimation"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(ekf=EKFStateEstimation)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for state estimation.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for state estimation.
	"""
	return IMPLEMENTATIONS


class StateEstimationMethodModel(CalibrationModel):
	"""The state estimation method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	replace_state_variables: bool = Field(
		description="Whether to replace or append the updated state variable",
		default=True,
	)
	stds: dict[str, float | list] | None = Field(
		description="The observation standard deviations", default=None
	)


class StateEstimationMethod(CalibrationMethodBase):
	"""The state estimation method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: StateEstimationMethodModel,
		engine: str = "ekf",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""StateEstimationMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (StateEstimationMethodModel): The calibration
				specification.
		    engine (str, optional): The state estimation backend.
				Defaults to "ekf".
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
