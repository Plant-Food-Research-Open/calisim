"""
Functional tests for the active learning module.

A battery of tests to validate the active learning calibration procedures.

"""

from calisim.active_learning import (
	ActiveLearningMethod,
	ActiveLearningMethodModel,
)
from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification

from ..conftest import get_calibrator


def test_skactiveml(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="gp",
		query_strategy="greedy_sampling_target",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(alpha=1e-10, optimizer="fmin_l_bfgs_b"),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None
