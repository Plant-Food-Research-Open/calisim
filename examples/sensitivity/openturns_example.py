import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.sensitivity import (
	SensitivityAnalysisMethod,
	SensitivityAnalysisMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = ParameterSpecification(
	parameters=[
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
)


def sensitivity_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)
	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values.mean()
	return simulated_data


outdir = get_examples_outdir()
specification = SensitivityAnalysisMethodModel(
	experiment_name="openturns_sensitivity_analysis",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	method="saltelli",
	order=4,
	n_samples=256,
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	calibration_func_kwargs=dict(t=observed_data.year),
)

calibrator = SensitivityAnalysisMethod(
	calibration_func=sensitivity_func, specification=specification, engine="openturns"
)

calibrator.specify().execute().analyze()

print(f"Results written to: {outdir}")
