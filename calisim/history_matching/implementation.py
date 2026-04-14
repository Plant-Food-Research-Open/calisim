"""Contains the implementations for the history matching methods

Implements the supported history matching methods.

"""

from collections.abc import Callable

import numpy as np
from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "history_matching"

BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	ies=f"calisim.{TASK}.ies_wrapper:IESHistoryMatching",
	pyesmda=f"calisim.{TASK}.pyesmda_wrapper:PyESMDAHistoryMatching",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for history matching.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for history matching.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


class HistoryMatchingMethodModel(CalibrationModel):
	"""The history matching method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	covariance: np.ndarray | None = Field(
		description="The covariance matrix for variables", default=None
	)


class HistoryMatchingMethod(CalibrationMethodBase):
	"""The history matching method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: HistoryMatchingMethodModel,
		engine: str = "ies",
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""HistoryMatchingMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (HistoryMatchingMethodModel): The calibration
				specification.
		    engine (str, optional): The history matching backend.
				Defaults to "ies".
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
