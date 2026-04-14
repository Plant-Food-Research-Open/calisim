"""Contains base classes for the various calibration methods

Abstract base classes are defined for the
simulation calibration procedures.

"""

import importlib
import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from importlib.metadata import entry_points
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

from ..data_model import (
	CalibrationModel,
	DistributionModel,
	ParameterDataType,
	ParameterEstimateModel,
	ParameterEstimatesModel,
	ParameterSpecification,
)
from ..statistics import get_full_factorial_design
from ..utils import (
	calibration_func_wrapper,
	create_file_path,
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
		self.artifacts: list[str] = []
		self.parameter_estimates = ParameterEstimatesModel(estimates=[])

		random_seed = self.specification.random_seed
		self.rng = np.random.default_rng(random_seed)
		self.simulation_ids: list[str] | None = []
		self.parameters: Any | None = None
		self.constants: dict[str, float | int | str | Any | None] = {}

		self._categoricals: dict[str, list[str]] = {}

	@abstractmethod
	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
		"""
		pass

	@abstractmethod
	def execute(self) -> None:
		"""Execute the simulation calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
		"""
		pass

	@abstractmethod
	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure.

		Raises:
		    NotImplementedError: Error raised for the unimplemented abstract method.
		"""
		pass

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

	def prepare_analyze(self) -> tuple[str, str, str, str | None]:
		"""Perform preparations for the analyze step.

		Returns:
		    tuple[str, str, str,  str | None]: A list of
		        metadata needed for the analysis outputs.
		"""
		task = self.task
		time_now = get_datetime_now()
		experiment_name = self.specification.experiment_name
		self.time_now = time_now
		outdir = self.specification.outdir
		return task, time_now, experiment_name, outdir  # type: ignore[return-value]

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

	def create_file_path(self, file_path: str) -> str:
		"""Create file path if it does not exist.

		Args:
		    file_path (str): The file path to create.

		Returns:
		    str: The created file path.
		"""
		return create_file_path(file_path)

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

	def get_full_factorial_design(
		self, parameter_spec: ParameterSpecification | None = None
	) -> np.ndarray:
		"""Get a full factorial design from a parameter specification.

		Args:
		    parameter_spec (ParameterSpecification | None, optional):
		        The simulation parameter specification. Defaults to None.

		Returns:
		    np.ndarray: The full factorial design.
		"""
		if parameter_spec is None:
			parameter_spec = self.specification.parameter_spec
		return get_full_factorial_design(parameter_spec)  # type: ignore[arg-type]

	def get_artifacts(self) -> list[str]:
		"""Getter method for the artifact list.

		Returns:
		    list[str]: The calibration workflow artifact list.
		"""
		return self.artifacts

	def append_artifact(self, artifact: str) -> None:
		"""Add a new artifact to the artifacts list.

		Args:
		    artifact (str): The artifact to append.
		"""
		self.artifacts.append(artifact)

	def present_fig(
		self,
		fig: Figure,
		outdir: str | None,
		time_now: str,
		task: str,
		experiment_name: str,
		suffix: str,
	) -> None:
		"""Present the figure by showing or writing to file.

		Args:
		    fig (Figure): The matplotlib figure.
		    outdir (str | None): The image output directory.
		    time_now (str): The current time.
		    task (str): The current calibration task.
		    suffix (str): The file name suffix.
		"""
		fig.tight_layout()
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-{suffix}.png"
			)
			self.append_artifact(outfile)
			fig.savefig(outfile)
		else:
			fig.show()

	def plot_simulated_vs_observed(
		self,
		simulated_data: np.ndarray,
		observed_data: np.ndarray,
		outdir: str,
		time_now: str,
		task: str,
		experiment_name: str,
		label: str = "",
	) -> None:
		"""Plot simulated data against observed data.

		Args:
		    simulated_data (np.ndarray): The simulated data.
		    observed_data (np.ndarray): The observed data.
		    outdir (str): The output directory.
		    time_now (str): The current time.
		    task (str): The calibration task.
		    experiment_name (str): The experiment name.
		    label (str, optional): The plot axes label. Defaults to "".
		"""
		simulated_label = "simulated"
		observed_label = "observed"

		if label != "":
			simulated_label += f" {label}"
			observed_label += f" {label}"

		df = pd.DataFrame(
			{simulated_label: simulated_data, observed_label: observed_data}
		)
		df["index"] = df.index

		fig, axes = plt.subplots(nrows=3, figsize=self.specification.figsize)
		df.plot.scatter("index", simulated_label, ax=axes[0])
		df.plot.scatter("index", observed_label, ax=axes[1])
		df.plot.scatter(simulated_label, observed_label, ax=axes[2])

		plot_suffix = f"{simulated_label}_vs_{observed_label}".replace(" ", "_")
		self.present_fig(fig, outdir, time_now, task, experiment_name, plot_suffix)

	def set_output_labels_from_Y(self, Y: np.ndarray) -> None:
		"""Set the simulation output labels from output data.

		Args:
		    Y (np.ndarray): The simulation outputs.
		"""
		output_labels = self.specification.output_labels

		if output_labels is None:
			if Y.ndim > 1:
				output_labels = [f"target_{i}" for i in range(Y.shape[1])]
			else:
				output_labels = ["target"]

		self.specification.output_labels = output_labels

	def get_parameter_estimates(self) -> ParameterEstimatesModel:
		"""Get the estimated parameter values, and potentially their uncertainties.

		Returns:
		    ParameterEstimatesModel: The estimated parameter values.
		"""
		return self.parameter_estimates

	def add_parameter_estimate(self, estimate: ParameterEstimateModel) -> None:
		"""Add a parameter estimate to the set of estimates.

		Args:
		    estimate (ParameterEstimateModel): The parameter estimate.
		"""
		self.parameter_estimates.estimates.append(estimate)

	def calculate_shap_importances(
		self,
		X: np.ndarray,
		emulator: BaseEstimator,
		names: list[str],
		test_size: float = 0,
		outfile: str | None = None,
	) -> None:
		"""Calculate SHAP importances using Kernel SHAP.

		Args:
		    X (np.ndarray): The training data.
		    emulator (BaseEstimator): The surrogate model.
		    names (list[str]): The parameter names.
		    test_size (float, optional): The test dataset size. Defaults to 0.
		    outfile (str | None, optional): The output file. Defaults to None.
		"""
		import shap
		from shap import KernelExplainer

		if test_size == 0:
			test_indx = 25
		else:
			test_indx = int(test_size * len(X))

		X_train = X[:test_indx]
		X_test = X[-test_indx:]

		explainer = KernelExplainer(emulator.predict, data=X_train, feature_names=names)
		shap_values = explainer.shap_values(X_test)

		show = False
		if outfile is None:
			show = True
		shap.summary_plot(shap_values, X_test, show=show, feature_names=names)

		if not show:
			self.append_artifact(outfile)  # type: ignore[arg-type]
			plt.tight_layout()
			plt.savefig(outfile)
			plt.close()

	def calc_uncertainty_calibration_metric(
		self,
		metric: str,
		mu: np.ndarray,
		sigma: np.ndarray,
		Y: np.ndarray,
		recal_model: BaseEstimator | None = None,
	) -> float:
		"""Calculate predictive uncertainty calibration metrics.

		Args:
		    metric (str): The metric name.
		    mu (np.ndarray): The conditional mean predictions.
		    sigma (np.ndarray): The conditional predicted standard deviations.
		    Y (np.ndarray): The simulation output data.
		    recal_model (BaseEstimator | None, optional): The prediction
		        recalibrator. Defaults to None.

		Returns:
		    float: The uncertainty calibration metric.
		"""
		import uncertainty_toolbox as uct

		metric_func = getattr(uct, metric)
		score = metric_func(mu, sigma, Y, recal_model=recal_model)
		return score

	def fit_recalibrator(
		self, emulator: BaseEstimator, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray
	) -> None:
		"""Fit a model recalibrator using Isotonic regression.

		Args:
		    emulator (BaseEstimator): The surrogate model.
		    mu (np.ndarray): The conditional mean predictions.
		    sigma (np.ndarray): The conditional predicted standard deviations.
		    Y (np.ndarray): The simulation output data.
		"""
		import uncertainty_toolbox as uct

		exp_props, obs_props = uct.get_proportion_lists_vectorized(mu, sigma, y)
		recal_model = uct.iso_recal(exp_props, obs_props)
		emulator.recal_model = recal_model

	def to_csv(self, df: pd.DataFrame, file_suffix: str) -> None:
		"""Convert dataframe to csv file.

		Args:
			df (pd.DataFrame): The dataframe.
			file_suffix (str): The file name suffix.
		"""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		outfile = self.join(
			outdir,  # type: ignore[arg-type]
			f"{time_now}-{task}-{experiment_name}-{file_suffix}.csv",
		)
		self.append_artifact(outfile)
		df.to_csv(outfile, index=False)

	def get_emulator(self) -> BaseEstimator:
		"""Get the trained emulator model.

		Returns:
			BaseEstimator: The emulator.
		"""
		return self.emulator

	def get_sampler(self) -> Any:
		"""Get the parameter sampler.

		Returns:
			Any: The sampler.
		"""
		return self.sampler

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Sample from the parameter space.

		Args:
			n_samples (int): The number of samples.

		Returns:
			np.ndarray: The sampled parameter values.
		"""
		raise NotImplementedError("Calibrator has not implemented sample_parameters()")

	def get_observed_data(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the observed data for calibration.

		Returns:
			np.ndarray | pd.DataFrame | None: The observed data for calibration.
		"""
		return self.specification.observed_data

	def get_X(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the parameter values.

		Returns:
			np.ndarray | pd.DataFrame | None: The parameter values.
		"""
		return self.X

	def get_Y(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the simulated data.

		Returns:
			np.ndarray | pd.DataFrame | None: The simulated data.
		"""
		return self.Y

	def get_simulation_ids(self) -> list[str] | None:
		"""Get the simulation IDs.

		Returns:
			list[str] | None: The simulation IDs.
		"""
		return self.simulation_ids

	def get_parameters(self) -> Any | None:
		"""Get the simulation parameters.

		Returns:
			Any | None: The simulation parameters.
		"""
		return self.parameters

	def get_constants(self) -> dict[str, float | int | str | Any | None] | None:
		"""Get the simulation constants.

		Returns:
			dict[str, float | int | str | Any | None] | None: The simulation constants.
		"""
		return self.constants

	def get_categorical_parameter(self, parameter_name: str, index: int) -> str | None:
		"""Get a categorical parameter value by its name and index.

		Args:
			parameter_name (str): The parameter name.
			index (int): The category index.

		Returns:
			str | None: The categorical parameter value.
		"""
		parameter_values = self._categoricals.get(parameter_name)
		if parameter_values is None:
			return None

		try:
			parameter_value = parameter_values[index]
		except IndexError:
			parameter_value = None
		return parameter_value

	def assign_categorical_parameter_values(
		self, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Assign categorical parameter values from an index value.

		Args:
			parameters (dict[str, Any]): The dictionary of parameter values.

		Returns:
			dict[str, Any]: The parameter values with assigned categorical values.
		"""
		for k in self._categoricals:
			index = parameters.get(k)
			if index is None:
				continue
			parameters[k] = self.get_categorical_parameter(k, index)
		return parameters

	def set_categorical_parameter(self, spec: DistributionModel) -> tuple[int, int]:
		"""Set and index a categorical parameter.

		Args:
			spec (DistributionModel): The parameter specification.

		Returns:
			tuple[int, int]: The categorical parameter bounds.
		"""
		parameter_name = spec.name
		parameter_values = spec.distribution_args
		self._categoricals[parameter_name] = parameter_values  # type: ignore[assignment]
		if parameter_values is None:
			return 0, 0
		else:
			return 0, len(parameter_values)


class CalibrationMethodBase(CalibrationWorkflowBase):
	"""The calibration method abstract class."""

	_PLUGIN_CACHE: dict[str, dict[str, str]] = {}
	_IMPLEMENTATION_CACHE: dict[str, type] = {}

	def __init__(
		self,
		calibration_func: Callable,
		specification: CalibrationModel,
		task: str,
		engine: str,
		implementations: dict[str, str],
		implementation: type[CalibrationWorkflowBase]
		| CalibrationWorkflowBase
		| None = None,
	) -> None:
		"""CalibrationMethodBase constructor.

		Args:
		    calibration_func (Callable): The calibration function.
		        For example, a simulation function or objective function.
		    specification (CalibrationModel): The calibration specification.
		    task (str): The calibration task.
		    engine (str): The calibration implementation engine.
		    implementations (dict[str, str]): The
		        list of supported engines.
		    implementation (type[CalibrationWorkflowBase] | CalibrationWorkflowBase
				| None): The calibration workflow implementation.
		"""
		super().__init__(calibration_func, specification, task)
		self.engine = engine
		self.supported_engines = list(implementations.keys())

		if implementation is None:
			if engine not in self.supported_engines:
				raise NotImplementedError(f"Unsupported {task} engine: {engine}")

			implementation_module = implementations.get(engine, None)
			if implementation_module is None:
				raise ValueError(
					f"{self.task} implementation not defined for: {engine}.",
					f"Supported engines are {', '.join(self.supported_engines)}",
				)

			implementation_class = self.resolve_implementation(implementation_module)
			self.implementation = implementation_class(
				calibration_func, specification, task
			)
		elif callable(implementation):
			self.implementation = implementation(calibration_func, specification, task)
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

	def resolve_implementation(
		self, implementation: str | type[CalibrationWorkflowBase]
	) -> type[CalibrationWorkflowBase]:
		"""Lazy load the implementation class from a module string.

		Args:
			implementation (str | type[CalibrationWorkflowBase]): The calibration
				implementation.

		Raises:
			ImportError: Error raised due to missing dependencies.

		Returns:
			type[CalibrationWorkflowBase]: The calibration workflow implementation.
		"""
		if isinstance(implementation, str):
			if implementation in self._IMPLEMENTATION_CACHE:
				return self._IMPLEMENTATION_CACHE[implementation]

			module_path, class_name = implementation.split(":")
			try:
				module = importlib.import_module(module_path)
			except ModuleNotFoundError as e:
				raise ImportError(
					f"Optional dependency for {implementation} is not installed"
				) from e
			implementation_class = getattr(module, class_name)
			self._IMPLEMENTATION_CACHE[implementation] = implementation_class
		else:
			implementation_class = implementation

		return implementation_class

	@classmethod
	def load_external_implementations(
		cls: type["CalibrationMethodBase"], task: str
	) -> dict[str, str]:
		"""Load external implementation plugins.

		Args:
			cls (type["CalibrationMethodBase"]): The calibration method class.
			task (str): The calibration task.

		Returns:
			dict[str, str]: The external implementation plugins.
		"""
		if task not in cls._PLUGIN_CACHE:
			implementations = entry_points().select(group=f"calisim.external.{task}")
			cls._PLUGIN_CACHE[task] = {
				implementation.name: implementation.value
				for implementation in implementations
			}
		return cls._PLUGIN_CACHE[task]

	def get_artifacts(self) -> list[str]:
		"""Getter method for the artifact list.

		Returns:
		    list[str]: The calibration workflow artifact list.
		"""
		return self.implementation.get_artifacts()

	def get_parameter_estimates(self) -> ParameterEstimatesModel:
		"""Get the estimated parameter values, and potentially their uncertainties.

		Returns:
		    ParameterEstimatesModel: The estimated parameter values.
		"""
		self._implementation_check("get_parameter_estimates")
		return self.implementation.get_parameter_estimates()

	def get_emulator(self) -> BaseEstimator:
		"""Get the trained emulator model.

		Returns:
			BaseEstimator: The emulator.
		"""
		self._implementation_check("get_emulator")
		return self.implementation.get_emulator()

	def get_sampler(self) -> Any:
		"""Get the parameter sampler.

		Returns:
			Any: The sampler.
		"""
		self._implementation_check("get_sampler")
		return self.implementation.get_sampler()

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Sample from the parameter space.

		Args:
			n_samples (int): The number of samples.

		Returns:
			np.ndarray: The sampled parameter values.
		"""
		self._implementation_check("get_sampler")
		return self.implementation.sample_parameters(n_samples)

	def get_observed_data(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the observed data for calibration.

		Returns:
			np.ndarray | pd.DataFrame | None: The observed data for calibration.
		"""
		self._implementation_check("get_observed_data")
		return self.implementation.get_observed_data()

	def get_X(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the parameter values.

		Returns:
			np.ndarray | pd.DataFrame | None: The parameter values.
		"""
		self._implementation_check("get_X")
		return self.implementation.get_X()

	def get_Y(self) -> np.ndarray | pd.DataFrame | None:
		"""Get the simulated data.

		Returns:
			np.ndarray | pd.DataFrame | None: The simulated data.
		"""
		self._implementation_check("get_Y")
		return self.implementation.get_Y()

	def get_simulation_ids(self) -> list[str] | None:
		"""Get the simulation IDs.

		Returns:
			list[str] | None: The simulation IDs.
		"""
		self._implementation_check("get_simulation_ids")
		return self.implementation.get_simulation_ids()

	def get_parameters(self) -> Any | None:
		"""Get the simulation parameters.

		Returns:
			Any | None: The simulation parameters.
		"""
		self._implementation_check("get_parameters")
		return self.implementation.get_parameters()

	def get_constants(self) -> dict[str, float | int | str | Any | None] | None:
		"""Get the simulation constants.

		Returns:
			dict[str, float | int | str | Any | None] | None: The simulation constants.
		"""
		self._implementation_check("get_constants")
		return self.implementation.get_constants()
