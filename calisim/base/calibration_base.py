"""Contains base classes for the various calibration methods

Abstract base classes are defined for the
simulation calibration procedures.

"""

import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..data_model import CalibrationModel, DistributionModel, ParameterDataType
from ..utils import (
	calibration_func_wrapper,
	extend_X,
	get_datetime_now,
	get_simulation_uuid,
)


def pre_post_hooks(f: Callable) -> Callable:
	"""Execute prehooks and posthooks for calibration methods.

	Args:
		f (Callable): The wrapped function.

	Returns:
		Callable: The wrapper function.
	"""

	@wraps(f)
	def wrapper(
		self: CalibrationWorkflowBase, *args: list, **kwargs: dict
	) -> "CalibrationWorkflowBase":
		"""The wrapper function for prehooks and posthooks.

		Returns:
			CalibrationWorkflowBase: The calibration workflow.
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
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (CalibrationModel): The calibration specification.
		    task (str): The calibration task.
		"""
		super().__init__()
		self.task = task
		self.calibration_func = calibration_func
		self.specification = specification

	@abstractmethod
	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("specify() method not implemented.")

	@abstractmethod
	def execute(self) -> None:
		"""Execute the simulation calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("execute() method not implemented.")

	@abstractmethod
	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
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
			tuple[str, str, str | None]: A list of
				metadata needed for the analysis outputs.
		"""
		task = self.task
		time_now = get_datetime_now()
		self.time_now = time_now
		outdir = self.specification.outdir
		return task, time_now, outdir

	def get_simulation_uuid(self) -> str:
		"""Get a new simulation uuid.

		Returns:
			str: The simulation uuid.
		"""
		return get_simulation_uuid()

	def extend_X(self, X: np.ndarray, Y_rows: int) -> np.ndarray:
		"""Extend the number of rows for X with a dummy index column.

		Args:
			X (np.ndarray): The input matrix.
			Y_rows (int) The number of rows for the simulation outputs.

		Returns:
			np.ndarray: The extended input matrix with a dummy column.
		"""
		return extend_X(X, Y_rows)

	def get_default_rng(self, random_seed: int | None = None) -> np.random.Generator:
		"""Get a numpy random number generator.

		Args:
			random_seed (int | None, optional): The
				random seed. Defaults to None.

		Returns:
			np.random.Generator: The random number generator.
		"""
		return np.random.default_rng(random_seed)

	def join(self, *paths: str) -> str:
		"""Join file paths.

		Args:
			paths (str): The file paths.

		Returns:
			str: The joined file paths.
		"""
		return osp.join(*paths)

	def get_parameter_bounds(self, spec: DistributionModel) -> tuple[float, float]:
		"""Get the lower and upper bounds from a parameter specification.

		Args:
			spec (DistributionModel): The parameter specification.

		Raises:
			ValueError: Error raised when the
				bounds cannot be identified.

		Returns:
			tuple[float, float]: The lower and upper bounds.
		"""
		distribution_args = spec.distribution_args
		if isinstance(distribution_args, list):
			if len(distribution_args) == 2:
				lower_bound, upper_bound = distribution_args
				return lower_bound, upper_bound

		distribution_kwargs = spec.distribution_kwargs
		if isinstance(distribution_kwargs, dict):
			lower_bound = distribution_kwargs.get("lower_bound", None)
			upper_bound = distribution_kwargs.get("upper_bound", None)
			if lower_bound is not None and upper_bound is not None:
				return lower_bound, upper_bound

		raise ValueError(f"Invalid parameter specification for {spec.name}")

	def get_calibration_func_kwargs(self) -> dict:
		"""Get the calibration function named arguments.

		Returns:
			dict: The calibration function named arguments.
		"""
		calibration_func_kwargs = self.specification.calibration_func_kwargs
		if calibration_func_kwargs is None:
			calibration_func_kwargs = {}

		pass_calibration_workflow = self.specification.pass_calibration_workflow
		if pass_calibration_workflow is not None:
			k = "calibration_workflow"
			if isinstance(pass_calibration_workflow, str):
				k = pass_calibration_workflow
			calibration_func_kwargs[k] = self

		return calibration_func_kwargs

	def prehook_calibration_func(
		self,
		parameters: dict | list[dict],
		simulation_id: str | list[str],
		observed_data: np.ndarray | None,
		**method_kwargs: dict,
	) -> tuple:
		"""Prehook to run before calling the calibration function

		Args:
			parameters (dict | List[dict]): The simulation parameters.
			simulation_id (str | List[str]): The simulation IDs.
			observed_data (np.ndarray | None): The observed data.

		Returns:
			tuple: The calibration function parameters.
		"""
		return parameters, simulation_id, observed_data, method_kwargs

	def posthook_calibration_func(
		self,
		results: np.ndarray | pd.DataFrame | float,
		parameters: dict | list[dict],
		simulation_id: str | list[str],
		observed_data: np.ndarray | None,
		**method_kwargs: dict,
	) -> tuple:
		"""Posthook to run after calling the calibration function

		Args:
			results (np.ndarray | pd.DataFrame | float): The simulation
				results.
			parameters (dict | List[dict]): The simulation parameters.
			simulation_id (str | List[str]): The simulation IDs.
			observed_data (np.ndarray | None): The observed data.

		Returns:
			tuple: The calibration function results and parameters.
		"""
		return results, parameters, simulation_id, observed_data, method_kwargs

	def call_calibration_func(
		self,
		parameters: dict | list[dict],
		simulation_id: str | list[str],
		observed_data: np.ndarray | None,
		**method_kwargs: dict,
	) -> float | list[float] | np.ndarray | pd.DataFrame:
		"""Wrapper method for the calibration function.

		Args:
			results (np.ndarray | pd.DataFrame | float): The simulation
				results.
			parameters (dict | List[dict]): The simulation parameters.
			simulation_id (str | List[str]): The simulation IDs.
			observed_data (np.ndarray | None): The observed data.

		Returns:
			float | list[float] | np.ndarray | pd.DataFrame: The
				calibration function results.
		"""
		prehook_results = self.prehook_calibration_func(
			parameters, simulation_id, observed_data, **method_kwargs
		)
		parameters, simulation_id, observed_data, method_kwargs = prehook_results

		results = self.calibration_func(
			parameters, simulation_id, observed_data, **method_kwargs
		)

		results, *_ = self.posthook_calibration_func(
			results, parameters, simulation_id, observed_data, **method_kwargs
		)
		return results

	def calibration_func_wrapper(
		self,
		X: np.ndarray,
		workflow: "CalibrationWorkflowBase",
		observed_data: pd.DataFrame | np.ndarray,
		parameter_names: list[str],
		data_types: list[ParameterDataType],
		calibration_kwargs: dict,
		wrap_values: bool = False,
	) -> np.ndarray:
		"""Wrapper function for the calibration function.

		Args:
			X (np.ndarray): The parameter set matrix.
			workflow (CalibrationWorkflowBase): The calibration workflow.
			observed_data (pd.DataFrame | np.ndarray): The observed data.
			parameter_names (list[str]): The list of simulation parameter names.
			data_types (list[ParameterDataType]): The data types for each parameter.
			calibration_kwargs (dict): Arguments to supply to the calibration function.
			wrap_values (bool): Whether to wrap scalar values with a list.
				Defaults to False.

		Returns:
			np.ndarray: The simulation output data.
		"""
		return calibration_func_wrapper(
			X,
			workflow,
			observed_data,
			parameter_names,
			data_types,
			calibration_kwargs,
			wrap_values,
		)

	def present_fig(
		self, fig: Figure, outdir: str | None, time_now: str, task: str, suffix: str
	) -> None:
		fig.tight_layout()
		if outdir is not None:
			outfile = self.join(outdir, f"{time_now}-{task}_{suffix}.png")
			fig.savefig(outfile)
		else:
			fig.show()


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
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (CalibrationModel): The calibration specification.
		    task (str): The calibration task.
		    engine (str): The calibration implementation engine.
		    implementations (dict[str, type[CalibrationWorkflowBase]]): The
				list of supported engines.
			implementation (CalibrationWorkflowBase | None): The
				calibration workflow implementation.
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
					f"{self.task} implementation not defined for: {engine}.",
					f"Supported engines are {', '.join(self.supported_engines)}",
				)
			self.implementation = implementation_class(
				calibration_func, specification, task
			)
		else:
			self.implementation = implementation

	def _implementation_check(self, function_name: str) -> None:
		"""Check that the implementation is set.

		Args:
			function_name (str): The name of the function.

		Raises:
			ValueError: Error raised when the implementation is not set.
		"""
		if self.implementation is None:
			raise ValueError(
				f"{self.task} implementation is not set when calling {function_name}()."
			)

	@pre_post_hooks
	def specify(self) -> "CalibrationMethodBase":
		"""Specify the parameters of the model calibration procedure.

		Raises:
			ValueError: Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase: The calibration method.
		"""
		self._implementation_check("specify")
		self.implementation.specify()
		return self

	@pre_post_hooks
	def execute(self) -> "CalibrationMethodBase":
		"""Execute the simulation calibration procedure.

		Raises:
			ValueError: Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase: The calibration method.
		"""
		self._implementation_check("execute")
		self.implementation.execute()
		return self

	@pre_post_hooks
	def analyze(self) -> "CalibrationMethodBase":
		"""Analyze the results of the simulation calibration procedure.

		Raises:
			ValueError: Error raised when the implementation is not set.

		Returns:
			CalibrationMethodBase: The calibration method.
		"""
		self._implementation_check("analyze")
		self.implementation.analyze()
		return self

	def get_engines(self, as_string: bool = False) -> list | str:
		"""Get a list of supported engines.

		Args:
		    as_string (bool, optional): Whether to return
				the engine list as a string. Defaults to False.

		Returns:
		    list | str: The list of supported engines.
		"""
		if as_string:
			return ", ".join(self.supported_engines)
		else:
			return self.supported_engines
