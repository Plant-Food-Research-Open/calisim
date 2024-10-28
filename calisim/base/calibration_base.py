"""Contains base classes for the various calibration methods

Abstract base classes are defined for the
simulation calibration procedures.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps

from ..data_model import CalibrationModel, DistributionModel
from ..utils import get_datetime_now


def pre_post_hooks(f: Callable) -> Callable:
	"""Execute prehooks and posthooks for calibration methods.

	Args:
		f (Callable):
			The wrapped function.

	Returns:
		Callable:
			The wrapper function.
	"""

	@wraps(f)
	def wrapper(
		self: CalibrationWorkflowBase, *args: list, **kwargs: dict
	) -> "CalibrationWorkflowBase":
		"""The wrapper function for prehooks and posthooks.

		Returns:
			CalibrationWorkflowBase:
				The calibration workflow.
		"""
		func_name = f.__name__
		getattr(self, f"prehook_{func_name}")()
		result = f(self, *args, **kwargs)
		getattr(self, f"posthook_{func_name}")()
		return result

	return wrapper


class CalibrationWorkflowBase(ABC):
	"""The calibration workflow abstract class."""

	def __init__(
		self, calibration_func: Callable, specification: CalibrationModel, task: str
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
			calibration_func (Callable):
				The calibration function.
				For example, a simulation function or objective function.
		    specification (CalibrationModel):
		        The calibration specification.
		    task (str):
		        The calibration task.
		"""
		super().__init__()
		self.task = task
		self.calibration_func = calibration_func
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

	def prehook_specify(self) -> None:
		"""Prehook to run before specify()."""
		pass

	def posthook_specify(self) -> None:
		"""Posthook to run after specify()."""
		pass

	def prehook_execute(self) -> None:
		"""Prehook to run before execute()."""
		pass

	def posthook_execute(self) -> None:
		"""Posthook to run after execute()."""
		pass

	def prehook_analyze(self) -> None:
		"""Prehook to run before analyze()."""
		pass

	def posthook_analyze(self) -> None:
		"""Posthook to run after analyze()."""
		pass

	def prepare_analyze(self) -> tuple[str, str, str | None]:
		"""Perform preparations for the analyze step.

		Returns:
			tuple[str, str, str | None]:
				A list of metadata needed for the analysis outputs.
		"""
		task = self.task
		time_now = get_datetime_now()
		self.time_now = time_now
		outdir = self.specification.outdir
		return task, time_now, outdir

	def get_parameter_bounds(self, spec: DistributionModel) -> tuple[float, float]:
		"""Get the lower and upper bounds from a parameter specification.

		Args:
			spec (DistributionModel):
				The parameter specification.

		Raises:
			ValueError:
				Error raised when the bounds cannot be identified.

		Returns:
			tuple[float, float]:
				The lower and upper bounds.
		"""
		dist_args = spec.dist_args
		if isinstance(dist_args, list):
			if len(dist_args) == 2:
				lower_bound, upper_bound = dist_args
				return lower_bound, upper_bound

		dist_kwargs = spec.dist_kwargs
		if isinstance(dist_kwargs, dict):
			lower_bound = dist_kwargs.get("lower_bound", None)
			upper_bound = dist_kwargs.get("upper_bound", None)
			if lower_bound is not None and upper_bound is not None:
				return lower_bound, upper_bound

		raise ValueError(
			f"Invalid parameter specification for {spec.name} of type {spec.dist_name}"
		)


class CalibrationMethodBase(CalibrationWorkflowBase):
	"""The calibration method abstract class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: CalibrationModel,
		task: str,
		engine: str,
		implementations: dict[str, type[CalibrationWorkflowBase]],
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
			calibration_func (Callable):
				The calibration function.
				For example, a simulation function or objective function.
		    specification (CalibrationModel):
		        The calibration specification.
		    task (str):
		        The calibration task.
		    engine (str):
		        The calibration implementation engine.
		    implementations (dict[str, type[CalibrationWorkflowBase]]):
		        The list of supported engines.
			implementation (CalibrationWorkflowBase | None):
				The calibration workflow implementation.
		"""
		super().__init__(calibration_func, specification, task)
		self.engine = engine
		self.supported_engines = list(implementations.keys())

		if implementation is None:
			if engine not in self.supported_engines:
				raise NotImplementedError(f"Unsupported {task} engine: {engine}")

			implementation_class = implementations.get(engine, None)
			if implementation_class is None:
				raise ValueError(
					f"{self.task} implementation not defined for: {engine}"
				)
			self.implementation = implementation_class(
				calibration_func, specification, task
			)
		else:
			self.implementation = implementation

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

	@pre_post_hooks
	def specify(self) -> "CalibrationMethodBase":
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

	@pre_post_hooks
	def execute(self) -> "CalibrationMethodBase":
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

	@pre_post_hooks
	def analyze(self) -> "CalibrationMethodBase":
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
