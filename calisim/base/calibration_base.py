"""Contains base classes for the various calibration methods

Abstract base classes are defined for the
simulation calibration procedures.

"""

from abc import ABC, abstractmethod

from ..data_model import DistributionCalibrationModel, IntervalCalibrationModel


class CalibrationWorkflowBase(ABC):
	"""The calibration workflow abstract class."""

	def __init__(
		self, specification: IntervalCalibrationModel | DistributionCalibrationModel
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
		    specification (IntervalCalibrationModel | DistributionCalibrationModel):
		        The calibration specification.
		"""
		super().__init__()
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
		specification: IntervalCalibrationModel | DistributionCalibrationModel,
		task: str,
		engine: str,
		implementations: dict[str, type[CalibrationWorkflowBase]],
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
		    specification (IntervalCalibrationModel | DistributionCalibrationModel):
		        The calibration specification.
		    task (str):
		        The calibration task.
		    engine (str):
		        The calibration implementation engine.
		    implementations (dict[str, type[CalibrationWorkflowBase]]):
		        The list of supported engines.
		"""
		super().__init__(specification)
		self.task = task

		self.engine = engine
		self.supported_engines = list(implementations.keys())
		if engine not in self.supported_engines:
			raise NotImplementedError(f"Unsupported {task} engine: {engine}")

		implementation = implementations.get(engine, None)
		if implementation is None:
			raise ValueError(f"{self.task} implementation not defined for: {engine}")
		self.implementation = implementation(specification)

	def _implementation_check(self, function_name: str) -> None:
		"""Check that the implementation is set.

		Args:
			function_name (str):
				The name of the function.

		Raises:
			ValueError:
				Error raised when the implementation is not set.
		"""
		if self.implementation is None:
			raise ValueError(
				f"{self.task} implementation is not set when calling {function_name}()."
			)

	def specify(self) -> "CalibrationMethodBase":  # type: ignore[override]
		"""Specify the parameters of the model calibration procedure.

		Raises:
			ValueError:
				Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase:
				The calibration method.
		"""
		self._implementation_check("specify")
		self.implementation.specify()
		return self

	def execute(self) -> "CalibrationMethodBase":  # type: ignore[override]
		"""Execute the simulation calibration procedure.

		Raises:
			ValueError:
				Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase:
				The calibration method.
		"""
		self._implementation_check("execute")
		self.implementation.execute()
		return self

	def analyze(self) -> "CalibrationMethodBase":  # type: ignore[override]
		"""Analyze the results of the simulation calibration procedure.

		Raises:
			ValueError:
				Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase:
				The calibration method.
		"""
		self._implementation_check("analyze")
		self.implementation.analyze()
		return self

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
