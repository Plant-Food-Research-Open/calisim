import pandas as pd

from calisim.data_model import DistributionModel, ParameterDataType
from calisim.example_models import LotkaVolterraModel
from calisim.sensitivity import (
	SensitivityAnalysisMethod,
	SensitivityAnalysisMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	DistributionModel(
		name="alpha",
		distribution_name="uniform",
		distribution_args=[0.45, 0.55],
		data_type=ParameterDataType.CONTINUOUS,
	),
	DistributionModel(
		name="beta",
		distribution_name="uniform",
		distribution_args=[0.02, 0.03],
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def sensitivity_func(
	parameters: dict, _: pd.DataFrame | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values.mean()
	return simulated_data


outdir = get_examples_outdir()
specification = SensitivityAnalysisMethodModel(
	experiment_name="salib_sensitivity_analysis",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	method="sobol",
	n_samples=128,
	output_labels=["Lynx"],
	verbose=True,
	vectorize=False,
	calibration_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(calc_second_order=True, scramble=True),
	analyze_kwargs=dict(
		calc_second_order=True,
		num_resamples=200,
		conf_level=0.95,
	),
)

calibrator = SensitivityAnalysisMethod(
	calibration_func=sensitivity_func, specification=specification, engine="salib"
)

calibrator.specify().execute().analyze()

print(f"Results written to: {outdir}")
