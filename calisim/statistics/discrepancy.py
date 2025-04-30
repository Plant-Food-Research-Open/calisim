"""Contains utilities for calculating discrepancy values.

This module defines utility functions for discrepancies
that can be used to compare simulated and observational data.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pydoc import locate

import numpy as np
import sklearn.metrics as metrics
from scipy.spatial import distance as sp_distance
from scipy.special import kl_div
from scipy.stats import (
	energy_distance,
	multivariate_normal,
	norm,
	poisson,
	wasserstein_distance,
)
from scipy.stats import t as t_dist


class DistanceMetricBase(ABC):
	"""The distance metric abstract class."""

	def __init__(self) -> None:
		"""DistanceMetricBase constructor."""
		super().__init__()

	@abstractmethod
	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.

		Raises:
		    NotImplementedError: Error raised for the
				unimplemented abstract method.
		"""
		raise NotImplementedError("calculate() method not implemented.")


class L1Norm(DistanceMetricBase):
	"""The L1 norm distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = np.linalg.norm(observed - simulated, ord=1)
		return distance


class L2Norm(DistanceMetricBase):
	"""The L2 norm distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = np.linalg.norm(observed - simulated, ord=2)
		return distance


class WassersteinDistance(DistanceMetricBase):
	"""The Wasserstein ID distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = wasserstein_distance(observed, simulated)
		return distance


class KlDivergence(DistanceMetricBase):
	"""The K-L divergence."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = kl_div(observed, simulated).sum()
		return distance


class EnergyDistance(DistanceMetricBase):
	"""The energy distance distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = energy_distance(observed, simulated)
		return distance


class BrayCurtisDistance(DistanceMetricBase):
	"""The Bray-Curtis distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.braycurtis(observed, simulated)
		return distance


class CanberraDistance(DistanceMetricBase):
	"""The Canberra distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.canberra(observed, simulated)
		return distance


class ChebyshevDistance(DistanceMetricBase):
	"""The Chebyshev distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.chebyshev(observed, simulated)
		return distance


class CorrelationDistance(DistanceMetricBase):
	"""The correlation distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.correlation(observed, simulated)
		return distance


class CosineDistance(DistanceMetricBase):
	"""The Cosine distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.cosine(observed, simulated)
		return distance


class MinkowskiDistance(DistanceMetricBase):
	"""The Minkowski distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.minkowski(observed, simulated)
		return distance


class JensenShannonDistance(DistanceMetricBase):
	"""The Jensen-Shannon distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = sp_distance.jensenshannon(observed, simulated)
		return distance


class MeanSquaredError(DistanceMetricBase):
	"""The mean squared error distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.mean_squared_error(observed, simulated)
		return distance


class MeanAbsoluteError(DistanceMetricBase):
	"""The mean absolute error distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.mean_absolute_error(observed, simulated)
		return distance


class RootMeanSquaredError(DistanceMetricBase):
	"""The root mean squared error distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.root_mean_squared_error(observed, simulated)
		return distance


class MeanPinballLoss(DistanceMetricBase):
	"""The mean pinball loss distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.mean_pinball_loss(observed, simulated)
		return distance


class MeanAbsolutePercentageError(DistanceMetricBase):
	"""The mean absolute percentage error distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.mean_absolute_percentage_error(observed, simulated)
		return distance


class MedianAbsoluteError(DistanceMetricBase):
	"""The median absolute error distance."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		distance = metrics.median_absolute_error(observed, simulated)
		return distance


def get_distance_metric_func(distance_metric: str) -> Callable:
	"""Get the distance metric function by name.

	Args:
	    distance_metric (str): The distance metric name.

	Returns:
	    Callable: The distance metric function.
	"""
	distance_metric = distance_metric.replace("_", " ").title().replace(" ", "")
	module = "calisim.statistics.distance_metrics"
	func: Callable = locate(f"{module}.{distance_metric}")  # type: ignore[assignment]
	return func


class GaussianLogLikelihood(DistanceMetricBase):
	"""The Gaussian log likelihood."""

	def calculate(
		self,
		observed: np.ndarray,
		simulated: np.ndarray,
		sigma: float | np.ndarray | None = None,
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
			sigma (float | np.ndarray | None, optional): The standard
				deviation. Defaults to None.
		"""
		if sigma is None:
			residuals = observed - simulated
			sigma = np.std(residuals, ddof=1)

		log_likelihood = norm.logpdf(observed, loc=simulated, scale=sigma).sum()
		return log_likelihood


class StudentsTLogLikelihood(DistanceMetricBase):
	"""The Student's t log likelihood."""

	def calculate(
		self,
		observed: np.ndarray,
		simulated: np.ndarray,
		nu: int = 1,
		sigma: float | np.ndarray | None = None,
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
			nu (int, optional): The degrees of freedom for the
				t distribution. Defaults to 1.
			sigma (float | np.ndarray | None, optional): The standard
				deviation. Defaults to None.
		"""
		if sigma is None:
			residuals = observed - simulated
			sigma = np.std(residuals, ddof=1)

		log_likelihood = t_dist.logpdf(
			observed, df=nu, loc=simulated, scale=sigma
		).sum()
		return log_likelihood


class PoissonLogLikelihood(DistanceMetricBase):
	"""The Poisson log likelihood."""

	def calculate(
		self, observed: np.ndarray, simulated: np.ndarray
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
		"""
		log_likelihood = poisson.logpmf(observed, mu=simulated).sum()
		return log_likelihood


class MultivariateNormalLogLikelihood(DistanceMetricBase):
	"""The multivariate normal log likelihood."""

	def calculate(
		self,
		observed: np.ndarray,
		simulated: np.ndarray,
		cov_matrix: np.ndarray | None = None,
		jitter: float = 1e-7,
	) -> float | np.ndarray:
		"""Calculate the distance between observed and simulated data.

		Args:
		    observed (np.ndarray): The observed data.
		    simulated (np.ndarray): The simulated data.
			cov_matrix (np.ndarray | None, optional): The covariance
				matrix. Defaults to None.
			jitter (float) Jitter to stabilise the inversion of
				the covariance matrix.
		"""
		if cov_matrix is None:
			residuals = observed - simulated
			cov_matrix = np.cov(residuals, rowvar=False, ddof=1)
			cov_matrix = cov_matrix + jitter * np.eye(cov_matrix.shape[0])

		log_likelihoods = [
			multivariate_normal.logpdf(obs, mean=pred, cov=cov_matrix)
			for obs, pred in zip(observed, simulated)
		]

		log_likelihood = np.sum(log_likelihoods)
		return log_likelihood
