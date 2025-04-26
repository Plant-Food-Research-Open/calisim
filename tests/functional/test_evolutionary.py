"""
Functional tests for the evolutionary module.

A battery of tests to validate the evolutionary calibration procedures.

"""

import pytest

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.evolutionary import EvolutionaryMethod, EvolutionaryMethodModel
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator, is_close

# def test_spotpy_dream(
# 	sir_model: ExampleModelContainer,
# 	sir_parameter_spec: ParameterSpecification,
# 	outdir: str,
# ) -> None:
# 	observed_data = sir_model.observed_data

# 	calibration_kwargs = dict(
# 		n_samples=250,
# 		method="dream",
# 		objective="gaussianLikelihoodMeasErrorOut",
# 		calibration_func_kwargs=dict(t=observed_data.day),
# 		method_kwargs=dict(nChains=4, nCr=3, delta=1),
# 	)

# 	calibrator = get_calibrator(
# 		EvolutionaryMethod,
# 		EvolutionaryMethodModel,
# 		sir_model,
# 		sir_parameter_spec,
# 		"spotpy",
# 		outdir,
# 		sir_model.output_labels,
# 		calibration_kwargs,
# 	)

# 	calibrator.specify().execute().analyze()
# 	assert is_close(sir_model, calibrator)


# def test_spotpy_abc(
# 	sir_model: ExampleModelContainer,
# 	sir_parameter_spec: ParameterSpecification,
# 	outdir: str,
# ) -> None:
# 	observed_data = sir_model.observed_data

# 	calibration_kwargs = dict(
# 		n_samples=250,
# 		method="abc",
# 		objective="rmse",
# 		calibration_func_kwargs=dict(t=observed_data.day),
# 		method_kwargs=dict(),
# 	)

# 	calibrator = get_calibrator(
# 		EvolutionaryMethod,
# 		EvolutionaryMethodModel,
# 		sir_model,
# 		sir_parameter_spec,
# 		"spotpy",
# 		outdir,
# 		sir_model.output_labels,
# 		calibration_kwargs,
# 	)

# 	calibrator.specify().execute().analyze()
# 	assert is_close(sir_model, calibrator)


# def test_spotpy_demcz(
# 	sir_model: ExampleModelContainer,
# 	sir_parameter_spec: ParameterSpecification,
# 	outdir: str,
# ) -> None:
# 	observed_data = sir_model.observed_data

# 	calibration_kwargs = dict(
# 		n_samples=250,
# 		method="demcz",
# 		objective="rmse",
# 		calibration_func_kwargs=dict(t=observed_data.day),
# 		method_kwargs=dict(),
# 	)

# 	calibrator = get_calibrator(
# 		EvolutionaryMethod,
# 		EvolutionaryMethodModel,
# 		sir_model,
# 		sir_parameter_spec,
# 		"spotpy",
# 		outdir,
# 		sir_model.output_labels,
# 		calibration_kwargs,
# 	)

# 	calibrator.specify().execute().analyze()
# 	assert is_close(sir_model, calibrator)


# def test_spotpy_fscabc(
# 	sir_model: ExampleModelContainer,
# 	sir_parameter_spec: ParameterSpecification,
# 	outdir: str,
# ) -> None:
# 	observed_data = sir_model.observed_data

# 	calibration_kwargs = dict(
# 		n_samples=250,
# 		method="fscabc",
# 		objective="rmse",
# 		calibration_func_kwargs=dict(t=observed_data.day),
# 		method_kwargs=dict(),
# 	)

# 	calibrator = get_calibrator(
# 		EvolutionaryMethod,
# 		EvolutionaryMethodModel,
# 		sir_model,
# 		sir_parameter_spec,
# 		"spotpy",
# 		outdir,
# 		sir_model.output_labels,
# 		calibration_kwargs,
# 	)

# 	calibrator.specify().execute().analyze()
# 	assert is_close(sir_model, calibrator)


# def test_spotpy_sceua(
# 	sir_model: ExampleModelContainer,
# 	sir_parameter_spec: ParameterSpecification,
# 	outdir: str,
# ) -> None:
# 	observed_data = sir_model.observed_data

# 	calibration_kwargs = dict(
# 		n_samples=250,
# 		method="sceua",
# 		objective="rmse",
# 		calibration_func_kwargs=dict(t=observed_data.day),
# 		method_kwargs=dict(),
# 	)

# 	calibrator = get_calibrator(
# 		EvolutionaryMethod,
# 		EvolutionaryMethodModel,
# 		sir_model,
# 		sir_parameter_spec,
# 		"spotpy",
# 		outdir,
# 		sir_model.output_labels,
# 		calibration_kwargs,
# 	)

# 	calibrator.specify().execute().analyze()
# 	assert is_close(sir_model, calibrator)


@pytest.mark.torch
def test_evotorch_ga(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_iterations=10,
		n_samples=10,
		method="ga",
		directions=["minimize"],
		operators=dict(
			OnePointCrossOver=dict(tournament_size=4),
			GaussianMutation=dict(stdev=0.1),
			PolynomialMutation=dict(eta=20.0),
		),
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(popsize=100, elitist=True, re_evaluate=True),
	)

	calibrator = get_calibrator(
		EvolutionaryMethod,
		EvolutionaryMethodModel,
		sir_model,
		sir_parameter_spec,
		"evotorch",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


@pytest.mark.torch
def test_evotorch_cmaes(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_iterations=10,
		n_samples=10,
		method="cmaes",
		directions=["minimize"],
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(stdev_init=1),
	)

	calibrator = get_calibrator(
		EvolutionaryMethod,
		EvolutionaryMethodModel,
		sir_model,
		sir_parameter_spec,
		"evotorch",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


@pytest.mark.torch
def test_evotorch_cosyne(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_iterations=10,
		n_samples=10,
		method="cosyne",
		directions=["minimize"],
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(popsize=100, tournament_size=20, mutation_stdev=1),
	)

	calibrator = get_calibrator(
		EvolutionaryMethod,
		EvolutionaryMethodModel,
		sir_model,
		sir_parameter_spec,
		"evotorch",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)
