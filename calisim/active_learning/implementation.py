"""Contains the implementations for active learning methods

Implements the supported active learning methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "active_learning"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	skactiveml=f"calisim.{TASK}.skactiveml_wrapper:SkActiveMLActiveLearning"
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for active learning.

	Returns:
		Dict[str, str]: The dictionary
			of calibration implementations for active learning.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


class ActiveLearningMethodModel(CalibrationModel):
	"""The active learning method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	query_strategy: str = Field(
		description="The active learning query strategy",
		default="greedy_sampling_target",
	)


class ActiveLearningMethod(CalibrationMethodBase):
	"""The active learning method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: ActiveLearningMethodModel,
		engine: str = "skactiveml",
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""ActiveLearningMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (ActiveLearningMethodModel): The calibration
				specification.
		    engine (str, optional): The active learning backend.
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
