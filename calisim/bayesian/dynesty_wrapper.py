"""Contains the implementations for Bayesian calibration methods using
dynesty

Implements the supported Bayesian calibration methods using
the dynesty library.

"""

import sys

import dynesty
import numpy as np
import pandas as pd
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType, ParameterEstimateModel


def prior_transform(utheta: list[float], self: CalibrationWorkflowBase) -> list[float]:
	transformed_theta = []
	for i, bounds in enumerate(self.bounds):
		x = utheta[i]
		lower_bound, upper_bound = bounds
		x = lower_bound + (upper_bound - lower_bound) * x
		data_type = self.data_types[i]
		if data_type == ParameterDataType.DISCRETE:
			x = int(x)
		transformed_theta.append(x)
	return transformed_theta


def target_function(
	X: np.ndarray, self: CalibrationWorkflowBase, bayesian_calibration_kwargs: dict
) -> np.ndarray:
	Y = self.calibration_func_wrapper(
		[X],
		self,
		self.specification.observed_data,
		self.names,
		self.data_types,
		bayesian_calibration_kwargs,
	)
	if np.isscalar(Y):
		return Y
	elif len(Y) == 1:
		return Y.item()
	else:
		raise ValueError(
			"Log likelihood for DynestyBayesianCalibration must be scalar value."
		)


class DynestyBayesianCalibration(CalibrationWorkflowBase):
	"""The dynesty Bayesian calibration method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.bounds = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			bounds = self.get_parameter_bounds(spec)
			self.bounds.append(bounds)

			data_type = spec.data_type
			self.data_types.append(data_type)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		sampler_name = self.specification.method
		supported_samplers = dict(
			dynamic=dynesty.DynamicNestedSampler,
			nested=dynesty.NestedSampler,
		)
		sampler_class = supported_samplers.get(sampler_name, None)
		if sampler_class is None:
			raise ValueError(
				f"Unsupported dynesty sampler: {sampler_name}.",
				f"Supported dynesty samplers are {', '.join(supported_samplers)}",
			)

		nlive = self.specification.n_init
		maxiter = self.specification.n_iterations
		if maxiter is None:
			maxiter = sys.maxsize
		maxcall = self.specification.n_samples
		if maxcall is None:
			maxcall = sys.maxsize

		random_seed = self.specification.random_seed
		rstate = np.random.default_rng(random_seed)
		n_jobs = self.specification.n_jobs

		bayesian_calibration_kwargs = self.get_calibration_func_kwargs()
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		if n_jobs > 1:
			with dynesty.pool.Pool(
				n_jobs,
				target_function,
				prior_transform,
				logl_args=(self, bayesian_calibration_kwargs),
				ptform_args=(self),
			) as pool:
				sampler = sampler_class(
					pool.loglike,
					pool.prior_transform,
					ndim=len(self.names),
					bound="multi",
					sample="auto",
					rstate=rstate,
					pool=pool,
					nlive=nlive,
				)
				sampler.run_nested(maxiter=maxiter, maxcall=maxcall, **method_kwargs)
		else:
			sampler = sampler_class(
				target_function,
				prior_transform,
				ndim=len(self.names),
				bound="multi",
				sample="auto",
				rstate=rstate,
				pool=None,
				nlive=nlive,
				logl_args=[self, bayesian_calibration_kwargs],
				ptform_args=[self],
			)
			sampler.run_nested(maxiter=maxiter, maxcall=maxcall, **method_kwargs)

		self.sampler = sampler

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		results = self.sampler.results
		for plot_func in [
			dyplot.traceplot,
			dyplot.cornerplot,
			dyplot.cornerpoints,
			dyplot.runplot,
		]:
			plot_name = plot_func.__name__.replace("_", "-")
			if plot_name == "runplot":
				plot_func(results)
			else:
				plot_func(results, labels=self.names)

			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-{plot_name}.png",
				)
				self.append_artifact(outfile)
				plt.tight_layout()
				plt.savefig(outfile)
				plt.close()
			else:
				plt.show()
				plt.close()

		samples, weights = results.samples, results.importance_weights()
		samples_equal = dyfunc.resample_equal(samples, weights)
		trace_df = pd.DataFrame(samples_equal, columns=self.names)

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

		if outdir is None:
			return

		self.to_csv(trace_df, "trace")
