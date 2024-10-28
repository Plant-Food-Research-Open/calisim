"""Contains the implementations for the history matching methods

Implements the supported history matching methods.

"""

from collections.abc import Callable

import numpy as np

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .ies_wrapper import IESHistoryMatching

TASK = "history_matching"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(ies=IESHistoryMatching)


def get_history_matching_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for history matching.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]:
			The dictionary of calibration implementations for history matching.
	"""
	return IMPLEMENTATIONS


class HistoryMatchingMethodModel(CalibrationModel):
	"""The history matching method data model.

	Args:
	    BaseModel (CalibrationModel):
	        The calibration base model class.
	"""

	covariance: np.ndarray | None = None


class HistoryMatchingMethod(CalibrationMethodBase):
	"""The history matching method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: HistoryMatchingMethodModel,
		engine: str = "ies",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""HistoryMatchingMethod constructor.

		Args:
			calibration_func (Callable):
				The calibration function.
				For example, a simulation function or objective function.
		    specification (HistoryMatchingMethodModel):
		        The calibration specification.
		    engine (str, optional):
		        The history matching backend. Defaults to "ies".
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
