"""
Functional tests for the quadrature module.

A battery of tests to validate the quadrature calibration procedures.

"""

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.quadrature import QuadratureMethod, QuadratureMethodModel

from ..conftest import get_calibrator


def test_emukit(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=5,
		n_iterations=10,
		n_samples=50,
		kernel="QuadratureRBFLebesgueMeasure",
		measure="LebesgueMeasure",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(noise_var=1e-4),
	)

	calibrator = get_calibrator(
		QuadratureMethod,
		QuadratureMethodModel,
		sir_model,
		sir_parameter_spec,
		"emukit",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None
