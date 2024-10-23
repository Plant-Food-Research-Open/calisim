"""Contains base classes for the various calibration methods

Abstract base classes are defined for the
simulation calibration procedures.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..data_model import CalibrationMethodModel


class CalibrationWorkflowBase(ABC):
	"""The calibration workflow abstract class."""

	def __init__(
		self, simulator: Callable, specification: CalibrationMethodModel
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
		    simulator (Callable):
		        The simulator function.
		    specification (CalibrationMethodModel):
		        The calibration specification.
		"""
		super().__init__()
		self.simulator = simulator
		self.specification = specification

	@abstractmethod
	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("specify() method not implemented.")

	@abstractmethod
	def execute(self) -> None:
		"""Execute the simulation calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("execute() method not implemented.")

	@abstractmethod
	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("analyze() method not implemented.")


class CalibrationMethodBase(CalibrationWorkflowBase):
	"""The calibration method abstract class."""

	def __init__(
		self,
		simulator: Callable,
		specification: CalibrationMethodModel,
		task: str,
		engine: str,
		supported_engines: list,
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
		    simulator (Callable):
		        The simulator function.
		    specification (CalibrationMethodModel):
		        The calibration specification.
		    task (str):
		        The calibration task.
		    engine (str):
		        The calibration implementation engine.
		    supported_engines (list):
		        The list of supported engines.
		"""
		super().__init__(simulator, specification)
		self.task = task
		self.engine = engine
		self.supported_engines = supported_engines
		self.implementation: CalibrationWorkflowBase | None = None

		if engine not in supported_engines:
			raise NotImplementedError(f"Unsupported {task} engine: {engine}")

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		if self.implementation is None:
			raise ValueError(
				"Optimisation implementation is not set when calling specify()."
			)
		self.implementation.specify()

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		if self.implementation is None:
			raise ValueError(
				"Optimisation implementation is not set when calling execute()."
			)
		self.implementation.execute()

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		if self.implementation is None:
			raise ValueError(
				"Optimisation implementation is not set when calling analyze()."
			)
		self.implementation.analyze()

	def get_engines(self, as_string: bool = False) -> list | str:
		"""Get a list of supported engines.

		Args:
		    as_string (bool, optional):
		        Whether to return the engine list as a string. Defaults to False.

		Returns:
		    list | str:
		        The list of supported engines.
		"""
		if as_string:
			return ", ".join(self.supported_engines)
		else:
			return self.supported_engines
