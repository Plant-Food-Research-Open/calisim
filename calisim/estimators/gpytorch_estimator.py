"""Contains the Scikit-Learn wrapper for GPyTorch

The defined Scikit-Learn wrapper for the GPyTorch library.

"""

import gpytorch
import torch
from skorch.probabilistic import ExactGPRegressor


class SingleTaskGPRegressionModel(gpytorch.models.ExactGP):
	"""The single task exact Gaussian process.

	Args:
	    gpytorch (ExactGP): The GPyTorch module.
	"""

	def __init__(
		self,
		likelihood: gpytorch.likelihoods.Likelihood,
		noise_init: float | None = None,
	) -> None:
		"""SingleTaskGPRegressionModel constructor.

		Args:
		    likelihood (gpytorch.likelihoods.Likelihood): The likelihood
		        for training the Gaussian process.
		    noise_init (float | None, optional): The
		        initial noise level. Defaults to None.
		"""
		super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, X: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		"""The forward pass.

		Args:
		    X (torch.Tensor): The input tensor.

		Returns:
		    gpytorch.distributions.MultivariateNormal: The predicted
		        mean and covariance.
		"""
		mean_x = self.mean_module(X)
		covar_x = self.covar_module(X)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_single_task_exact_gp(
	lr: float, max_epochs: int, device: str = "cpu"
) -> ExactGPRegressor:
	"""Get an instance of a single task Gaussian process.

	Args:
	    lr (float): The learning rate.
	    max_epochs (int): The maximum number of epochs.
	    device (str, optional): The device to train the model. Defaults to "cpu".

	Returns:
	    ExactGPRegressor: The Gaussian process.
	"""
	return ExactGPRegressor(
		SingleTaskGPRegressionModel,
		optimizer=torch.optim.Adam,
		criterion=gpytorch.mlls.ExactMarginalLogLikelihood,
		lr=lr,
		max_epochs=max_epochs,
		device=device,
		batch_size=-1,
	)
