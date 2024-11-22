"""Contains the implementations for uncertainty analysis methods using
OpenTurns

Implements the supported uncertainty analysis methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType


class OpenTurnsUncertaintyAnalysis(CalibrationWorkflowBase):
	"""The OpenTurns uncertainty analysis method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			dist_instance = getattr(ot, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters.append(parameter)

		distribution_collection = ot.DistributionCollection(parameters)
		self.parameters = ot.JointDistribution(distribution_collection)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		observed_data = self.specification.observed_data
		parameter_names = self.names
		data_types = self.data_types
		uncertainty_kwargs = self.get_calibration_func_kwargs()

		def uncertainty_func(X: np.ndarray) -> np.ndarray:
			parameters = []
			for theta in X:
				parameter_set = {}
				for i, parameter_value in enumerate(theta):
					parameter_name = parameter_names[i]
					data_type = data_types[i]
					if data_type == ParameterDataType.CONTINUOUS:
						parameter_set[parameter_name] = parameter_value
					else:
						parameter_set[parameter_name] = int(parameter_value)
				parameters.append(parameter_set)

			simulation_ids = [
				self.get_simulation_uuid() for _ in range(len(parameters))
			]

			if self.specification.batched:
				results = self.call_calibration_func(
					parameters, simulation_ids, observed_data, **uncertainty_kwargs
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.call_calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**uncertainty_kwargs,
					)
					results.append(result)  # type: ignore[arg-type]

			results = np.array(results)
			return results

		n_samples = self.specification.n_samples
		experiment = ot.LHSExperiment(self.parameters, n_samples)
		X = self.specification.X
		if X is None:
			X = experiment.generate()

		n_dim = self.parameters.getDimension()
		n_out = self.specification.n_out
		Y = self.specification.Y
		if Y is None:
			uncertainty_func_wrapper = ot.PythonFunction(
				n_dim, n_out, func_sample=uncertainty_func
			)
			Y = uncertainty_func_wrapper(X)

		emulator_name = self.specification.solver
		emulators = dict(
			functional_chaos=ot.FunctionalChaosAlgorithm, kriging=ot.KrigingAlgorithm
		)
		emulator_class = emulators.get(emulator_name, None)
		if emulator_class is None:
			raise ValueError(
				f"Unsupported emulator: {emulator_name}.",
				f"Supported emulators are {', '.join(emulators)}",
			)

		emulator = emulator_class(X, Y)
		emulator.run()

		self.emulator = emulator

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		pass
		# task, time_now, outdir = self.prepare_analyze()
		# solver_name = self.specification.solver
