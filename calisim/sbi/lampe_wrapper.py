"""Contains the implementations for simulation-based inference methods using
LAMPE

Implements the supported simulation-based inference methods using
the LAMPE library.

"""

import os.path as osp
from itertools import islice

import numpy as np
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

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


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


class LAMPESimulationBasedInference(CalibrationWorkflowBase):
	"""The LAMPE simulation-based inference method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		parameter_spec = self.specification.parameter_spec.parameters

		self.names = []
		priors = []
		for spec in parameter_spec:
			name = spec.name
			self.names.append(name)

			data_type = spec.data_type
			if data_type == ParameterDataType.DISCRETE:
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				lower_bound = np.floor(lower_bound).astype("int")
				upper_bound = np.floor(upper_bound).astype("int")
				replicates = np.floor(upper_bound - lower_bound).astype("int")
				probabilities = torch.tensor([1 / replicates])
				probabilities = probabilities.repeat(replicates)
				base_distribution = dist.Categorical(probabilities)
				transforms = [
					dist.AffineTransform(
						loc=torch.Tensor([lower_bound]), scale=torch.Tensor([1])
					)
				]
				prior = dist.TransformedDistribution(base_distribution, transforms)
			else:
				distribution_name = (
					spec.distribution_name.replace("_", " ").title().replace(" ", "")
				)

				distribution_args = spec.distribution_args
				if distribution_args is None:
					distribution_args = []
				distribution_args = [torch.Tensor([arg]) for arg in distribution_args]

				distribution_kwargs = spec.distribution_kwargs
				if distribution_kwargs is None:
					distribution_kwargs = {}
				distribution_kwargs = {
					k: torch.Tensor([v]) for k, v in distribution_kwargs.items()
				}

				distribution_class = getattr(dist, distribution_name)

				prior = distribution_class(*distribution_args, **distribution_kwargs)

			priors.append(prior)
			self.parameters = PriorCollection(priors)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		def simulator_func(theta: np.ndarray) -> np.ndarray:
			theta = theta.detach().cpu().numpy().T
			parameters = {}
			for i, name in enumerate(self.names):
				parameters[name] = theta[i]

			sbi_kwargs = self.get_calibration_func_kwargs()

			observed_data = self.specification.observed_data
			simulation_id = get_simulation_uuid()
			results = self.call_calibration_func(
				parameters, simulation_id, observed_data, **sbi_kwargs
			)
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
		step = GDStep(optimizer, clip=0.0)
		estimator.train()

		for epoch in range(self.specification.n_iterations):
			for theta, x in islice(loader, self.specification.num_simulations):
				neg_log_p = loss(theta, x)
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
		task, time_now, outdir = self.prepare_analyze()

		x_star = torch.from_numpy(self.specification.observed_data).float()
		n_draws = self.specification.n_samples
		with torch.no_grad():
			posterior_samples = self.estimator.flow(x_star).sample((n_draws,))

		for plot_func in [analysis.pairplot, analysis.marginal_plot]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=self.names)
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
				fig.savefig(outfile)
			else:
				fig.show()

		n_simulations = self.specification.num_simulations
		levels, coverages = expected_coverage_mc(
			posterior=self.estimator.flow,
			pairs=((theta, x) for theta, x in islice(self.loader, n_simulations)),
		)

		fig = coverage_plot(levels, coverages, legend=task)
		if outdir is not None:
			outfile = osp.join(
				outdir,
				f"{time_now}-{coverage_plot.__name__}.png",
			)
			fig.savefig(outfile)
		else:
			fig.show()
