"""Contains the implementations for evolutionary algorithms using SPOTPY

Implements the supported evolutionary algorithms using the SPOTPY library.

"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import spotpy.analyser as analyser
import spotpy.likelihoods as likelihoods
import spotpy.objectivefunctions as objectivefunctions
import spotpy.parameter
from spotpy.algorithms import NSGAII, abc, demcz, dream, fscabc, sceua
from spotpy.parameter import Base, generate

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterEstimateModel


class SPOTSetup:
	"""The SPOTPY calibration setup."""

	def __init__(
		self,
		workflow: CalibrationWorkflowBase,
		objective_function: Callable,
		evolutionary_name: str,
	):
		"""SPOTSetup constructor.

		Args:
			workflow (CalibrationWorkflowBase): The calibration workflow object.
			objective_function (Callable): The simulation objective function.
			evolutionary_name (str): The name of the evolutionary algorithm.
		"""
		self.objective_function = objective_function
		self.evolutionary_name = evolutionary_name
		self.setup_from_workflow(workflow)

	def setup_from_workflow(self, workflow: CalibrationWorkflowBase) -> None:
		"""Configure the calibration procedure from the workflow object.

		Args:
			workflow (CalibrationWorkflowBase): The calibration workflow object.
		"""
		parameter_names = []
		data_types = []
		priors: list[Base] = []

		parameter_spec = workflow.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			parameter_names.append(parameter_name)
			data_type = spec.data_type
			data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)
			distribution_class = getattr(spotpy.parameter, distribution_name)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			prior = distribution_class(
				parameter_name, *distribution_args, **distribution_kwargs
			)
			priors.append(prior)

		self.parameter_names = parameter_names
		self.data_types = data_types
		self.priors = priors
		self.call_calibration_func = workflow.call_calibration_func
		self.observed_data = workflow.specification.observed_data
		self.evolutionary_kwargs = workflow.get_calibration_func_kwargs()
		self.workflow = workflow

	def parameters(self) -> np.ndarray:
		"""Generate parameters from the prior specification.

		Returns:
			np.ndarray: The generated parameters.
		"""
		return generate(self.priors)

	def simulation(self, X: np.ndarray) -> np.ndarray:
		"""Run the simulation.

		Args:
			X (np.ndarray): The simulation parameter vector.

		Returns:
			np.ndarray: The simulation results.
		"""
		X = [X]
		results = self.workflow.calibration_func_wrapper(
			X,
			self.workflow,
			self.observed_data,
			self.parameter_names,
			self.data_types,
			self.evolutionary_kwargs,
		)
		return results[0]

	def evaluation(self) -> np.ndarray | pd.DataFrame:
		"""Get the observed data.

		Returns:
			np.ndarray | pd.DataFrame: The observed data.
		"""
		return self.observed_data

	def objectivefunction(
		self,
		simulation: np.ndarray | pd.DataFrame,
		evaluation: np.ndarray | pd.DataFrame,
	) -> float:
		"""Call the objective function on simulated and observed data.

		Args:
			simulation (np.ndarray | pd.DataFrame): The simulated data.
			evaluation (np.ndarray | pd.DataFrame): The observed data.

		Returns:
			float: The objective function results.
		"""
		if self.evolutionary_name == "dream":
			objective = self.objective_function(evaluation, simulation)
		else:
			objective = -self.objective_function(evaluation, simulation)
		return objective


class SPOTPYEvolutionary(CalibrationWorkflowBase):
	"""The SPOTPY evolutionary algorithm method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		evolutionary_name = self.specification.method
		objective = self.specification.objective

		if evolutionary_name == "dream":
			objective_function = getattr(likelihoods, objective)
		else:
			objective_function = getattr(objectivefunctions, objective)

		self.spot_setup = SPOTSetup(self, objective_function, evolutionary_name)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		evolutionary_name = self.specification.method
		evolutionary_algorithms = dict(
			abc=abc, demcz=demcz, dream=dream, fscabc=fscabc, nsgaii=NSGAII, sceua=sceua
		)
		evolutionary_class = evolutionary_algorithms.get(evolutionary_name, None)
		if evolutionary_class is None:
			raise ValueError(
				f"Unsupported evolutionary algorithm: {evolutionary_name}.",
				f"Supported algorithms are {', '.join(evolutionary_algorithms.keys())}",
			)

		_, time_now, dbname, outdir = self.prepare_analyze()
		if outdir is None:
			dbformat = "ram"
		else:
			dbformat = "csv"
			dbname = self.join(outdir, f"{time_now}_{dbname}")

		self.sampler = evolutionary_class(
			spot_setup=self.spot_setup,
			dbname=dbname,
			dbformat=dbformat,
			parallel="seq",
			save_sim=True,
		)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}
		n_samples = self.specification.n_samples
		self.sample_results = self.sampler.sample(n_samples, **method_kwargs)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		results = self.sampler.getdata()

		if outdir is None:
			return

		for plot_func in [
			analyser.plot_fast_sensitivity,
			analyser.plot_parametertrace,
			analyser.plot_parameterInteraction,
		]:
			plot_func_name = plot_func.__name__
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}_{plot_func_name}.png"
			)
			self.append_artifact(outfile)
			plot_func(results, fig_name=outfile)

		evaluation = self.specification.observed_data
		for plot_func in [
			analyser.plot_objectivefunction,
			analyser.plot_regression,
		]:
			plot_func_name = plot_func.__name__
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}_{plot_func_name}.png"
			)
			self.append_artifact(outfile)
			plot_func(results, evaluation, fig_name=outfile)

		if self.specification.method == "dream":
			plot_func = analyser.plot_gelman_rubin
			plot_func_name = plot_func.__name__
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}_{plot_func_name}.png"
			)
			self.append_artifact(outfile)
			plot_func(results, self.sample_results, outfile)

		for col in results.dtype.names:
			if not col.startswith("par"):
				continue
			values = np.array(results[col])
			name = col.replace("par", "")
			estimate = values.mean()
			uncertainty = values.std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)
