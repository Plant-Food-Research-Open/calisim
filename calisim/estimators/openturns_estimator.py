"""Contains the Scikit-Learn wrappers for OpenTurns

The defined Scikit-Learn wrappers for the OpenTurns library.

"""

import copy

import numpy as np
import openturns as ot
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FunctionalChaosEstimator(MultiOutputMixin, RegressorMixin, BaseEstimator):
	def __init__(self, parameters: ot.JointDistribution, total_degree: int = 2):
		"""FunctionalChaosEstimator constructor.

		Args:
		    parameters (dict): The distribution of parameters.
		    total_degree (int, optional): The total polynomial degree.
		        Defaults to 2.
		"""
		self.parameters = parameters

		input_dim = parameters.getDimension()
		columns = [
			ot.StandardDistributionPolynomialFactory(self.parameters.getMarginal(i))
			for i in range(input_dim)
		]
		enumerate_func = ot.LinearEnumerateFunction(input_dim)
		product_basic = ot.OrthogonalProductPolynomialFactory(columns, enumerate_func)
		candidate_basis = enumerate_func.getBasisSizeFromTotalDegree(total_degree)
		self.adaptive_strategy = ot.FixedStrategy(product_basic, candidate_basis)
		self.projection_strategy = ot.LeastSquaresStrategy()

		self.emulator = None

	def fit(
		self, X: np.ndarray, y: np.ndarray | None = None
	) -> "FunctionalChaosEstimator":
		"""Fit the estimator.

		Args:
		    X (np.ndarray): The simulation inputs.
		    y (np.ndarray | None, optional): The simulation outputs.
		        Defaults to None.

		Returns:
		    FunctionalChaosEstimator: The estimator.
		"""
		X = np.array(X)
		y = np.array(y)
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

		self.algo = ot.FunctionalChaosAlgorithm(
			X, y, self.parameters, self.adaptive_strategy, self.projection_strategy
		)
		self.algo.run()
		self.result = self.algo.getResult()
		self.emulator = self.result.getMetaModel()

		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Make a prediction.

		Args:
		    X (np.ndarray): The simulation inputs.

		Returns:
		    np.ndarray: The model predictions.
		"""
		check_is_fitted(self, "is_fitted_")
		X = check_array(X, ensure_2d=True, dtype="numeric")

		y_pred = self.emulator(X)
		return np.array(y_pred)

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
		y_pred = self.emulator(X)
		return r2_score(y_pred, y, sample_weight=sample_weight)


class KrigingEstimator(MultiOutputMixin, RegressorMixin, BaseEstimator):
	def __init__(
		self,
		parameters: ot.JointDistribution,
		basis: str = "constant",
		covariance: str = "SquaredExponential",
		covariance_scale: float = 1.0,
		covariance_amplitude: float = 1.0,
		n_out: int = 1,
	):
		"""KrigingEstimator constructor.

		Args:
		    parameters (dict): The distribution of parameters.
		    basis (str, optional): The basis function.
		        Defaults to "constant".
		    covariance (str, optional): The covariance function.
		        Defaults to "SquaredExponential".
		    covariance_scale (float, optional): Scale coefficient.
		        Defaults to 1.0.
		    covariance_amplitude (float, optional): Amplitude of the process.
		        Defaults to 1.0.
		    n_out (int, optional): The number of outputs.
		"""
		self.parameters = parameters
		input_dim = parameters.getDimension()

		supported_basis = dict(
			constant=ot.ConstantBasisFactory,
			linear=ot.LinearBasisFactory,
			quadratic=ot.QuadraticBasisFactory,
		)
		self.basis = supported_basis.get(basis, ot.ConstantBasisFactory)(
			input_dim
		).build()

		if n_out > 1:
			self.basis = ot.Basis(
				[
					ot.AggregatedFunction([self.basis.build(k)] * n_out)
					for k in range(self.basis.getSize())
				]
			)

		self.covariance = getattr(ot, covariance)(
			[covariance_scale] * input_dim, [covariance_amplitude]
		)

		if n_out > 1:
			self.covariance = ot.TensorizedCovarianceModel(
				[copy.deepcopy(self.covariance) for _ in range(n_out)]
			)

		self.emulator = None

	def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KrigingEstimator":
		"""Fit the estimator.

		Args:
		    X (np.ndarray): The simulation inputs.
		    y (np.ndarray | None, optional): The simulation outputs.
		        Defaults to None.

		Returns:
		    KrigingEstimator: The estimator.
		"""
		X = np.array(X)
		y = np.array(y)
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

		self.algo = ot.KrigingAlgorithm(X, y, self.covariance, self.basis)
		self.algo.run()
		self.result = self.algo.getResult()
		self.emulator = self.result.getMetaModel()

		return self

	def predict(
		self,
		X: np.ndarray,
	) -> np.ndarray | tuple:
		"""Make a prediction.

		Args:
		    X (np.ndarray): The simulation inputs.

		Returns:
		    np.ndarray | tuple: The model predictions.
		"""
		check_is_fitted(self, "is_fitted_")
		X = check_array(X, ensure_2d=True, dtype="numeric")
		y_pred = self.emulator(X)
		return np.array(y_pred)

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
		y_pred = self.emulator(X)
		return r2_score(y_pred, y, sample_weight=sample_weight)
