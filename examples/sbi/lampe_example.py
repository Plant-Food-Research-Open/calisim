import numpy as np
import pandas as pd

from calisim.data_model import DistributionModel, ParameterDataType
from calisim.example_models import LotkaVolterraModel
from calisim.sbi import (
	SimulationBasedInferenceMethod,
	SimulationBasedInferenceMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	DistributionModel(
		name="alpha",
		distribution_name="normal",
		distribution_args=[0.4, 0.03],
		data_type=ParameterDataType.CONTINUOUS,
	),
	DistributionModel(
		name="beta",
		distribution_name="normal",
		distribution_args=[0.025, 0.003],
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def sbi_func(
	parameters: dict, _: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


outdir = get_examples_outdir()
specification = SimulationBasedInferenceMethodModel(
	experiment_name="lampe_simulation_based_inference",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_samples=100,
	n_iterations=100,
	epsilon=0.1,
	output_labels=["Lynx"],
	verbose=True,
	vectorize=False,
	calibration_func_kwargs=dict(t=observed_data.year),
)

calibrator = SimulationBasedInferenceMethod(
	calibration_func=sbi_func, specification=specification, engine="lampe"
)

calibrator.specify().execute().analyze()

print(f"Results written to: {outdir}")
