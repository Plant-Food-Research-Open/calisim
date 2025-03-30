"""Contains the implementations for the likelihood-free methods

Implements the supported likelihood-free methods.

"""

import importlib
from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "likelihood_free"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict()

if importlib.util.find_spec("elfi") is not None:
	from .elfi_wrapper import ELFILikelihoodFree

	IMPLEMENTATIONS["elfi"] = ELFILikelihoodFree


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for likelihood-free.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for likelihood-free.
	"""
	return IMPLEMENTATIONS


class LikelihoodFreeMethodModel(CalibrationModel):
	"""The likelihood-free method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	epsilon: float = Field(
		description="The dissimilarity threshold between observed and simulated data",
		default=0,
	)
	acq_noise_var: float = Field(
		description="Noise added to Lower Confidence Bound Selection Criterion",
		default=0,
	)
	sampler: str = Field(
		description="The sampling algorithm to perform inference", default="metropolis"
	)


class LikelihoodFreeMethod(CalibrationMethodBase):
	"""The likelihood-free method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: LikelihoodFreeMethodModel,
		engine: str = "elfi",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""LikelihoodFreeMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (LikelihoodFreeMethodModel): The
				calibration specification.
		    engine (str, optional): The likelihood-free backend.
				Defaults to "elfi".
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
