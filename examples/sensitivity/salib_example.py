import pandas as pd

from calisim.data_model import ParameterDataType, ParameterIntervalModel
from calisim.example_models import LotkaVolterraModel
from calisim.sensitivity import (
	SensitivityAnalysisMethod,
	SensitivityAnalysisMethodModel,
)
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	ParameterIntervalModel(
		name="alpha",
		lower_bound=0.45,
		upper_bound=0.55,
		data_type=ParameterDataType.CONTINUOUS,
	),
	ParameterIntervalModel(
		name="beta",
		lower_bound=0.02,
		upper_bound=0.03,
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def sensitivity_func(
	parameters: dict, observed_data: pd.DataFrame | None
) -> float | list[float]:
	simulation_parameters = dict(
		h0=34.0, l0=5.9, t=observed_data.year, gamma=0.84, delta=0.026
	)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters)
	observed_data = observed_data.drop(columns=["year"])

	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy


specification = SensitivityAnalysisMethodModel(
	experiment_name="salib_sensitivity_analysis",
	parameter_spec=parameter_spec,
	observed_data=observed_data,
	outdir=get_examples_outdir(),
	sampler="sobol",
	n_samples=8,
	output_labels=["Discrepancy"],
	verbose=True,
	sampler_kwargs=dict(
		calc_second_order=True,
		num_resamples=200,
		conf_level=0.95,
	),
)

calibrator = SensitivityAnalysisMethod(
	calibration_func=sensitivity_func, specification=specification, engine="salib"
)

calibrator.specify().execute().analyze()
