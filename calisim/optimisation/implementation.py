"""Contains the implementations for the optimisation methods

Implements the supported optimisation methods.

"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..base import CalibrationMethodBase
from ..data_model import IntervalCalibrationModel
from .botorch_wrapper import BoTorchOptimisation
from .optuna_wrapper import OptunaOptimisation


class OptimisationMethodModel(IntervalCalibrationModel):
	"""The optimisation method data model.

	Args:
	    BaseModel (IntervalCalibrationModel):
	        The calibration base model class.
	"""

	objective: Callable
	observed_data: np.ndarray | pd.DataFrame
	directions: list[str] | None = ["minimize"]
	sampler: str
	sampler_kwargs: dict[str, Any] | None = None
	optimisation_kwargs: dict[str, Any] | None = None
	objective_kwargs: dict[str, Any] | None = None


class OptimisationMethod(CalibrationMethodBase):
	"""The optimisation method class."""

	def __init__(
		self,
		specification: OptimisationMethodModel,
		engine: str = "optuna",
	) -> None:
		"""OptimisationMethod constructor.

		Args:
		    specification (OptimisationMethodModel):
		        The calibration specification.
		    engine (str, optional):
		        The optimisation backend. Defaults to "optuna".
		"""
		task = "optimisation"

		implementations = dict(optuna=OptunaOptimisation, botorch=BoTorchOptimisation)

		super().__init__(specification, task, engine, implementations)
