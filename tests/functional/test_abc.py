"""
Functional tests for the ABC module.

A battery of tests to validate the ABC calibration procedures.

"""

from calisim.abc import (
	ApproximateBayesianComputationMethod,
	ApproximateBayesianComputationMethodModel,
)
from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator, is_close


def test_pyabc(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=25,
		walltime=3,
		epsilon=0.1,
		n_bootstrap=15,
		min_population_size=5,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(
			max_total_nr_simulations=500, max_nr_populations=20, min_acceptance_rate=0.0
		),
	)

	calibrator = get_calibrator(
		ApproximateBayesianComputationMethod,
		ApproximateBayesianComputationMethodModel,
		sir_model,
		sir_parameter_spec,
		"pyabc",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_pymc(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_samples=100,
		n_chains=1,
		n_jobs=1,
		epsilon=0.05,
		sum_stat="identity",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(compute_convergence_checks=True, return_inferencedata=True),
	)

	calibrator = get_calibrator(
		ApproximateBayesianComputationMethod,
		ApproximateBayesianComputationMethodModel,
		sir_model,
		sir_parameter_spec,
		"pymc",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator, rtol=0.3)
