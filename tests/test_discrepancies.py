"""
Tests for various discrepancy metrics and functions.

A battery of tests to validate the discrepancy metrics and functions.

"""

from calisim.base import ExampleModelContainer
from calisim.statistics import (
	BrayCurtisDistance,
	CanberraDistance,
	ChebyshevDistance,
	CorrelationDistance,
	CosineDistance,
	EnergyDistance,
	GaussianLogLikelihood,
	JensenShannonDistance,
	KlDivergence,
	L1Norm,
	L2Norm,
	MeanAbsoluteError,
	MeanAbsolutePercentageError,
	MeanPinballLoss,
	MeanSquaredError,
	MedianAbsoluteError,
	MinkowskiDistance,
	MultivariateNormalLogLikelihood,
	PoissonLogLikelihood,
	RootMeanSquaredError,
	StudentsTLogLikelihood,
	WassersteinDistance,
)


def test_bracy_curtis_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = BrayCurtisDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_bracy_curtis_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = BrayCurtisDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_canberra_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = CanberraDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_canberra_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = CanberraDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_chebyshev_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = ChebyshevDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_chebyshev_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = ChebyshevDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_correlation_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = CorrelationDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_correlation_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = CorrelationDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_cosine_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = CosineDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_cosine_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = CosineDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_energy_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = EnergyDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_energy_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = EnergyDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_gaussian_ll_lt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = GaussianLogLikelihood()
	assert metric.calculate(observed_data, simulated_data) < 0


def test_jensen_shannon_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = JensenShannonDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_jensen_shannon_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = JensenShannonDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_kl_divergence_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = KlDivergence()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_kl_divergence_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = KlDivergence()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_l1_norm_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = L1Norm()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_l1_norm_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = L1Norm()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_l2_norm_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = L2Norm()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_l2_norm_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = L2Norm()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_mae_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MeanAbsoluteError()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_mae_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MeanAbsoluteError()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_mape_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MeanAbsolutePercentageError()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_mape_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MeanAbsolutePercentageError()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_pinball_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MeanPinballLoss()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_pinball_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MeanPinballLoss()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_mse_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MeanSquaredError()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_mse_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MeanSquaredError()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_median_absolute_error_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MedianAbsoluteError()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_median_absolute_error_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MedianAbsoluteError()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_minkowski_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = MinkowskiDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_minkowski_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = MinkowskiDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_rmse_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = RootMeanSquaredError()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_rmse_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = RootMeanSquaredError()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_wasserstein_gt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = WassersteinDistance()
	assert metric.calculate(observed_data, simulated_data) > 0


def test_wasserstein_same(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten()
	metric = WassersteinDistance()
	assert metric.calculate(observed_data, simulated_data) == 0


def test_students_t_lt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = StudentsTLogLikelihood()
	assert metric.calculate(observed_data, simulated_data, nu=2) < 0


def test_poisson_lt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values.flatten()
	simulated_data = sir_model.observed_data.values.flatten() * 1.1
	metric = PoissonLogLikelihood()
	assert metric.calculate(observed_data, simulated_data) < 0


def test_mvn_lt(sir_model: ExampleModelContainer) -> None:
	observed_data = sir_model.observed_data.values
	simulated_data = sir_model.observed_data.values * 1.1
	metric = MultivariateNormalLogLikelihood()
	assert metric.calculate(observed_data, simulated_data, jitter=1e-3) < 0
