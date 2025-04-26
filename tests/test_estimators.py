"""
Tests for various scikit-learn estimators.

A battery of tests to validate scikit-learn surrogate model estimators.

"""

from calisim.base import ExampleModelContainer
from calisim.estimators import EmukitEstimator


def test_emukit_fit(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data
	col_names = observed_data.columns
	X_cols = col_names[:-1]
	y_col = col_names[-1]
	X = observed_data[X_cols].values
	y = observed_data[y_col].values

	estimator = EmukitEstimator()
	estimator.fit(X, y)
	estimator.predict(X)

	assert estimator.is_fitted_
