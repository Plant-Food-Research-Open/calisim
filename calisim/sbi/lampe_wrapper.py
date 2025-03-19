"""Contains the implementations for simulation-based inference methods using
LAMPE

Implements the supported simulation-based inference methods using
the LAMPE library.

"""

from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.optim as optim
from lampe.data import JointLoader
from lampe.diagnostics import expected_coverage_mc
from lampe.inference import NPE, NPELoss
from lampe.plots import coverage_plot
from lampe.utils import GDStep
from matplotlib import pyplot as plt
from sbi import analysis as analysis

from ..base import SimulationBasedInferenceBase
from ..data_model import DistributionModel, ParameterDataType, ParameterEstimateModel


class PriorCollection:
	"""A wrapper around a collection of priors."""

	def __init__(self, priors: list[dist.Distribution]) -> None:
		"""PriorCollection constructor.

		Args:
		    priors (list[dist.Distribution]): The list of prior distributions.
		"""
		self.parameters = priors

	def sample(self, batch_shape: tuple = ()) -> torch.Tensor:
		"""Sample from the priors.

		Args:
		    batch_shape (tuple, optional): The batch shape of
				the sampled priors. Defaults to ().

		Returns:
		    torch.Tensor: The sampled priors.
		"""
		prior_sample = []
		for prior in self.parameters:
			prior_sample.append(prior.sample(batch_shape).squeeze())
		return torch.stack(prior_sample).T


class LAMPESimulationBasedInference(SimulationBasedInferenceBase):
	"""The LAMPE simulation-based inference method class."""

	def preprocess(
		self, theta: torch.Tensor, parameter_spec: list[DistributionModel] | Any | None
	) -> torch.Tensor:
		"""Normalise the parameters of the simulation.
		Args:
		    theta (torch.Tensor): The simulation parameters.
		    parameter_spec (list[DistributionModel] | Any | None):
				The parameter specification.
		Raises:
		    ValueError: Error raised when an unsupported distribution is provided.
		Returns:
		    torch.Tensor: The normalised parameters.
		"""
		param_values = []
		for i, spec in enumerate(parameter_spec):  # type: ignore[arg-type]
			x = theta[i]
			distribution_name = spec.distribution_name
			if (
				distribution_name == "uniform"
				or spec.data_type == ParameterDataType.DISCRETE
			):
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				param_value = 2 * (x - lower_bound) / (upper_bound - lower_bound) - 1
			elif distribution_name == "normal":
				mu, sd = self.get_parameter_bounds(spec)
				param_value = (x - mu) / sd
			else:
				raise ValueError(
					f"Unsupported distribution for LAMPE: {distribution_name}"
				)
			param_values.append(param_value)
		return torch.Tensor(param_values)

	def postprocess(
		self,
		samples: torch.Tensor,
		parameter_spec: list[DistributionModel] | Any | None,
	) -> torch.Tensor:
		"""Reverse normalise the parameters of the simulation.
		Args:
		    samples (torch.Tensor): The normalised parameters.
		    parameter_spec (list[DistributionModel] | Any | None):
				The parameter specification.
		Raises:
		    ValueError: Error raised when an unsupported distribution is provided.
		Returns:
		    torch.Tensor: The denormalised parameters.
		"""
		param_values = []
		for sample in samples:
			norm_param_values = []
			for i, spec in enumerate(parameter_spec):  # type: ignore[arg-type]
				x = sample[i]
				distribution_name = spec.distribution_name
				if (
					distribution_name == "uniform"
					or spec.data_type == ParameterDataType.DISCRETE
				):
					lower_bound, upper_bound = self.get_parameter_bounds(spec)
					param_value = (x + 1) / 2 * (
						upper_bound - lower_bound
					) + lower_bound
				elif distribution_name == "normal":
					mu, sd = self.get_parameter_bounds(spec)
					param_value = x * sd + mu
				else:
					raise ValueError(
						f"Unsupported distribution for LAMPE: {distribution_name}"
					)
				norm_param_values.append(param_value)
			param_values.append(norm_param_values)
		return torch.Tensor(param_values)

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		super().specify()
		self.parameters: PriorCollection = PriorCollection(self.parameters)  # type: ignore[assignment]

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		sbi_kwargs = self.get_calibration_func_kwargs()

		def simulator_func(X: np.ndarray) -> np.ndarray:
			X = X.detach().cpu().numpy().T
			X = [X]
			results = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				sbi_kwargs,
			)
			results = results[0]
			return torch.from_numpy(results).float()

		loader = JointLoader(
			self.parameters, simulator_func, batch_size=1, vectorized=True
		)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		estimator = NPE(
			len(self.names),
			len(self.specification.observed_data),  # type: ignore[arg-type]
			**method_kwargs,
		)
		loss = NPELoss(estimator)
		optimizer = optim.Adam(estimator.parameters(), lr=self.specification.lr)
		step = GDStep(optimizer, clip=1.0)
		estimator.train()

		parameter_spec = self.specification.parameter_spec.parameters
		for epoch in range(self.specification.n_iterations):
			for theta, x in islice(loader, self.specification.num_simulations):
				neg_log_p = loss(self.preprocess(theta, parameter_spec), x)
				step(neg_log_p)
			if self.specification.verbose:
				print(f"Epoch {epoch + 1} : Negative log-likelihood {neg_log_p}")  # type: ignore[possibly-undefined]

		self.loader = loader
		self.estimator = estimator
		self.loss = loss
		self.optimizer = optimizer
		self.step = step

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		parameter_spec = self.specification.parameter_spec.parameters
		x_star = torch.from_numpy(self.specification.observed_data).float()
		n_draws = self.specification.n_samples
		with torch.no_grad():
			posterior_samples = self.estimator.flow(x_star).sample((n_draws,))
			posterior_samples = self.postprocess(posterior_samples, parameter_spec)

		for plot_func in [analysis.pairplot, analysis.marginal_plot]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=self.names)
			self.present_fig(
				fig, outdir, time_now, task, experiment_name, plot_func.__name__
			)

		n_simulations = self.specification.num_simulations
		levels, coverages = expected_coverage_mc(
			posterior=self.estimator.flow,
			pairs=((theta, x) for theta, x in islice(self.loader, n_simulations)),
		)

		fig = coverage_plot(levels, coverages, legend=task)
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, coverage_plot.__name__
		)

		trace_df = pd.DataFrame(
			posterior_samples.cpu().detach().numpy(), columns=self.names
		)
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trace.csv")  # type: ignore[arg-type]
		self.append_artifact(outfile)
		trace_df.to_csv(outfile, index=False)

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)
