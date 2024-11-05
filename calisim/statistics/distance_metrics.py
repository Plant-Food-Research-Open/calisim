"""Contains utilities for calculating distance values.

This module defines utility functions for distance metrics
that can be used to compare simulated and observational data.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pydoc import locate

import numpy as np
import sklearn.metrics as metrics


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


def get_distance_metrics() -> list[dict]:
	"""Get a list of available distance metrics and labels.

	Returns:
	    list[dict]: A list of available distance metrics and labels.
	"""
	distance_metrics: list[str] = [
		"l1_norm",
		"mean_squared_error",
		"mean_absolute_error",
		"root_mean_squared_error",
		"mean_pinball_loss",
		"mean_absolute_percentage_error",
		"median_absolute_error",
	]

	distance_metric_list = []
	for distance_metric in distance_metrics:
		label = distance_metric.replace("_", " ").title()
		distance_metric_list.append({"value": distance_metric, "label": label})

	return distance_metric_list
