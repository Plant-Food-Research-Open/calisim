"""Contains the Scikit-Learn wrapper for Emukit

The defined Scikit-Learn wrapper for the Emukit library.

"""

import numpy as np
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class EmukitEstimator(MultiOutputMixin, RegressorMixin, BaseEstimator):
	def __init__(self, method_kwargs: dict | None = None):
		"""EmukitEstimator constructor.

		Args:
		    method_kwargs (dict, optional): The named arguments for the estimator.
		        Defaults to None.
		"""
		if method_kwargs is None:
			method_kwargs = {}
		self.method_kwargs = method_kwargs
		self.emulator = None

	def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "EmukitEstimator":
		"""Fit the estimator.

		Args:
		    X (np.ndarray): The simulation inputs.
		    y (np.ndarray | None, optional): The simulation outputs.
		        Defaults to None.

		Returns:
		    EmukitEstimator: The estimator.
		"""
		if len(y.shape) == 1:
			y = np.expand_dims(y, axis=1)

		X, y = check_X_y(
			X,
			y,
			multi_output=True,
			y_numeric=True,
			ensure_2d=True,
			dtype="numeric",
		)
		self.is_fitted_ = True
		self.n_features_in_ = X.shape[1]

		gp = GPRegression(X, y, **self.method_kwargs)
		self.emulator = GPyModelWrapper(gp)

		return self

	def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray | tuple:
		"""Make a prediction.

		Args:
		    X (np.ndarray): The simulation inputs.
		    return_std (bool, optional): Whether to return the standard
				deviation. Defaults to False.

		Returns:
		    np.ndarray | tuple: The model predictions.
		"""
		check_is_fitted(self, "is_fitted_")
		X = check_array(X, ensure_2d=True, dtype="numeric")

		y_mu, y_var = self.emulator.predict(X)

		if return_std:
			return y_mu, np.sqrt(y_var)
		else:
			return y_mu

	def score(
		self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
	) -> float:
		"""Assess the estimator.

		Args:
		    X (np.ndarray): The simulation inputs.
		    y (np.ndarray): The simulation outputs.
		    sample_weight (np.ndarray | None, optional): Weighting factor
		        for the samples. Defaults to None.

		Returns:
		    float: The assessment score.
		"""
		y_pred = self.emulator.predict(X)
		return r2_score(y_pred, y, sample_weight=sample_weight)
