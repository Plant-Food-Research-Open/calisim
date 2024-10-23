"""Contains the implementations for the optimisation methods

Implements the supported optimisation methods.

"""

from collections.abc import Callable

from ..base import CalibrationMethodBase
from ..data_model import CalibrationMethodModel
from .botorch_wrapper import BoTorchOptimisation
from .optuna_wrapper import OptunaOptimisation


class OptimisationMethod(CalibrationMethodBase):
	"""The optimisation method class."""

	def __init__(
		self,
		simulator: Callable,
		specification: CalibrationMethodModel,
		engine: str = "optuna",
	) -> None:
		"""OptimisationMethod constructor.

		Args:
		    simulator (Callable):
		        The simulator function.
		    specification (CalibrationMethodModel):
		        The calibration specification.
		    engine (str, optional):
		        The optimisation backend. Defaults to "optuna".
		"""
		task = "optimisation"
		supported_engines = ["optuna", "botorch"]

		super().__init__(simulator, specification, task, engine, supported_engines)

		if engine == "optuna":
			self.implementation = OptunaOptimisation(simulator, specification)
		elif engine == "botorch":
			self.implementation = BoTorchOptimisation(simulator, specification)
