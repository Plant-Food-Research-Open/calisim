"""Contains the implementations for uncertainty analysis methods using
Pygpc

Implements the supported uncertainty analysis methods using
the Pygpc library.

"""

import os.path as osp
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import pygpc
from matplotlib import pyplot as plt
from pygpc.AbstractModel import AbstractModel

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class PygpcModel(AbstractModel):
	def __init__(
		self,
		workflow: CalibrationWorkflowBase,
		parameter_names: list[str],
		data_types: list[str],
	):
		super(type(self), self).__init__(matlab_model=False)
		self.calibration_func = workflow.calibration_func
		self.observed_data = workflow.specification.observed_data
		self.batched = workflow.specification.batched

		uncertainty_kwargs = workflow.specification.calibration_func_kwargs
		if uncertainty_kwargs is None:
			uncertainty_kwargs = {}
		self.uncertainty_kwargs = uncertainty_kwargs

		self.parameter_names = parameter_names
		self.data_types = data_types

	def validate(self) -> None:
		pass

	def simulate(
		self, process_id: int | None = None, matlab_engine: Callable | None = None
	) -> np.ndarray:
		parameter_name = self.parameter_names[0]
		N = self.p[parameter_name].shape[0]

		parameters = []
		for i in range(N):
			parameter_set = {}
			for j, parameter_name in enumerate(self.parameter_names):
				parameter_value = self.p[parameter_name][i]
				data_type = self.data_types[j]
				if data_type == ParameterDataType.CONTINUOUS:  # type: ignore[comparison-overlap]
					parameter_set[parameter_name] = parameter_value
				else:
					parameter_set[parameter_name] = int(parameter_value)
			parameters.append(parameter_set)

		simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]

		if self.batched:
			results = self.calibration_func(
				parameters,
				simulation_ids,
				self.observed_data,
				**self.uncertainty_kwargs,
			)
		else:
			results = []
			for i, parameter in enumerate(parameters):
				simulation_id = simulation_ids[i]
				result = self.calibration_func(
					parameter,
					simulation_id,
					self.observed_data,
					**self.uncertainty_kwargs,
				)
				results.append(result)
		results = np.array(results)

		if len(results.shape) == 1:
			results = results[:, np.newaxis]
		return results


class PygpcUncertaintyAnalysis(CalibrationWorkflowBase):
	"""The Pygpc uncertainty analysis method class."""

	def dist_name_processing(self, name: str) -> str:
		"""Apply data preprocessing to the distribution name.

		Args:
		    name (str): The unprocessed distribution name.

		Returns:
		    str: The processed distribution name.
		"""
		name = name.replace("_", " ").title().replace(" ", "")

		if name == "Normal":
			name = "Norm"
		return name

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		names = []
		data_types = []
		parameters = OrderedDict()

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			names.append(parameter_name)

			data_type = spec.data_type
			data_types.append(data_type)

			distribution_name = self.dist_name_processing(spec.distribution_name)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			if len(distribution_kwargs.keys()) == 0 and len(distribution_args) == 2:
				distribution_args = [distribution_args]

			dist_instance = getattr(pygpc, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters[parameter_name] = parameter

		self.parameters = parameters
		self.model = PygpcModel(self, names, data_types)  # type: ignore[arg-type]
		self.problem = pygpc.Problem(self.model, parameters)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		_, time_now, outdir = self.prepare_analyze()
		if outdir is None:
			fn_results = None
		else:
			experiment_name = self.specification.experiment_name
			fn_results = osp.join(outdir, f"{time_now}_{experiment_name}")

		options = self.specification.method_kwargs
		if options is None:
			options = {}

		options["method"] = self.specification.method
		options["solver"] = self.specification.solver
		options["n_cpu"] = self.specification.n_jobs
		options["n_grid"] = self.specification.n_init
		options["grid_options"] = dict(seed=self.specification.random_seed)
		options["grid"] = pygpc.Random
		options["fn_results"] = fn_results
		options["verbose"] = self.specification.verbose

		algorithm_name = self.specification.algorithm
		algorithms = dict(
			static_io=pygpc.Static_IO,
			static=pygpc.Static,
			me_static=pygpc.MEStatic,
			me_static_io=pygpc.MEStatic_IO,
			static_projection=pygpc.StaticProjection,
			me_static_projection=pygpc.MEStaticProjection,
			reg_adaptive=pygpc.RegAdaptive,
			me_reg_adaptive_projection=pygpc.MERegAdaptiveProjection,
			reg_adaptive_projection=pygpc.RegAdaptiveProjection,
		)
		algorithm_class = algorithms.get(algorithm_name, None)
		if algorithm_class is None:
			raise ValueError(
				f"Unsupported Pygpc algorithm: {algorithm_name}.",
				f"Supported Pygpc algorithm are {', '.join(algorithms)}",
			)
		algorithm = algorithm_class(problem=self.problem, options=options)

		session = pygpc.Session(algorithm=algorithm)
		session, coeffs, results = session.run()

		self.session = session
		self.coeffs = coeffs
		self.results = results
		self.options = options

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()
		n_samples = self.specification.n_samples
		outfile = None

		output_label = self.specification.output_labels[0]  # type: ignore[index]
		observed_data = self.specification.observed_data
		X = np.arange(0, observed_data.shape[-1], 1)
		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)
		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		for i in range(self.results.shape[0]):
			axes[1].plot(X, self.results[i])
		axes[1].set_title(f"Ensemble {output_label}")

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_ensemble_{output_label}.png")
			fig.savefig(outfile)
		else:
			fig.show()

		plot_func = pygpc.validate_gpc_mc
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}_{task}_{plot_func.__name__}")
		plot_func(
			session=self.session,
			coeffs=self.coeffs,
			fn_out=outfile,
			n_cpu=self.session.n_cpu,
		)
		if outdir is None:
			plt.show()
			plt.close()

		plot_func = pygpc.validate_gpc_plot
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}_{task}_{plot_func.__name__}")
		plot_func(
			session=self.session,
			coeffs=self.coeffs,
			random_vars=self.model.parameter_names,
			fn_out=outfile,
			n_grid=[n_samples, n_samples],
			n_cpu=self.session.n_cpu,
		)
		if outdir is None:
			plt.show()
			plt.close()
