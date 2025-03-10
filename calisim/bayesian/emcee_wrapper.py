"""Contains the implementations for Bayesian calibration methods using
emcee

Implements the supported Bayesian calibration methods using
the emcee library.

"""

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterEstimateModel


class EmceeBayesianCalibration(CalibrationWorkflowBase):
	"""The emcee Bayesian calibration method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.bounds: tuple[list[float], list[float]] = ([], [])

		parameter_spec = self.specification.parameter_spec.parameters
		n_walkers = self.specification.n_samples
		self.rng = self.get_default_rng(self.specification.random_seed)

		self.parameters = []
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			bounds = self.get_parameter_bounds(spec)
			lower_bound, upper_bound = bounds
			lower_bounds, upper_bounds = self.bounds
			lower_bounds.append(lower_bound)
			upper_bounds.append(upper_bound)

			distribution_name = spec.distribution_name.replace(" ", "_").lower()

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}
			distribution_args = list(distribution_kwargs.values())

			dist_instance = getattr(self.rng, distribution_name)
			samples = dist_instance(*distribution_args, size=n_walkers)
			self.parameters.append(samples)

		self.parameters = np.array(self.parameters).T

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		bayesian_calibration_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			lower_bounds, upper_bounds = self.bounds

			for i, x in enumerate(X):
				if x < lower_bounds[i] or x > upper_bounds[i]:
					return -np.inf

			X = np.array([X])
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				bayesian_calibration_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)

			return Y

		def log_target_function(X: np.ndarray) -> np.ndarray:
			Y = target_function(X)
			if Y == -np.inf:
				return Y
			return np.log(Y)

		n_walkers, n_dim = self.parameters.shape

		if self.specification.log_density:
			log_probability = log_target_function
		else:
			log_probability = target_function

		supported_moves = dict(
			RedBlueMove=emcee.moves.RedBlueMove,
			StretchMove=emcee.moves.StretchMove,
			WalkMove=emcee.moves.WalkMove,
			KDEMove=emcee.moves.KDEMove,
			DEMove=emcee.moves.DEMove,
			DESnookerMove=emcee.moves.DESnookerMove,
			MHMove=emcee.moves.MHMove,
			GaussianMove=emcee.moves.GaussianMove,
		)

		moves_spec = self.specification.moves
		if moves_spec is None:
			moves = None
		else:
			moves = []
			for name, weight in moves_spec.items():
				move = supported_moves.get(name)
				if move is None:
					raise ValueError(
						f"Unsupported move: {name}.",
						f"Supported moves are {', '.join(supported_moves)}",
					)
				moves.append((move, weight))

		self.sampler = emcee.EnsembleSampler(
			n_walkers, n_dim, log_probability, moves=moves, vectorize=False
		)

		n_iterations = self.specification.n_iterations
		verbose = self.specification.verbose
		self.sampler.run_mcmc(
			self.parameters, n_iterations, tune=True, progress=verbose
		)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		samples = self.sampler.get_chain()

		labels = self.names
		n_dim = len(labels)
		fig, axes = plt.subplots(n_dim, figsize=self.specification.figsize, sharex=True)
		for i in range(n_dim):
			ax = axes[i]
			ax.plot(samples[:, :, i], "k", alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(labels[i])
			ax.yaxis.set_label_coords(-0.1, 0.5)

		axes[-1].set_xlabel("Step number")
		self.present_fig(fig, outdir, time_now, task, experiment_name, "trace")

		flat_samples = self.sampler.get_chain(flat=True)
		fig = corner.corner(flat_samples, labels=labels)
		self.present_fig(fig, outdir, time_now, task, experiment_name, "plot_surface")

		trace_df = pd.DataFrame(flat_samples, columns=labels)
		fig, axes = plt.subplots(nrows=n_dim, figsize=self.specification.figsize)

		for i, parameter_name in enumerate(labels):
			axes[i].set_title(parameter_name)
			axes[i].hist(trace_df[parameter_name], label=parameter_name)

		self.present_fig(fig, outdir, time_now, task, experiment_name, "plot_slice")

		if outdir is None:
			return

		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trace.csv")
		self.append_artifact(outfile)
		trace_df.to_csv(outfile, index=False)

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)
