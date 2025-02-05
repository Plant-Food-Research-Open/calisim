import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.history_matching import (
	HistoryMatchingMethod,
	HistoryMatchingMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = ParameterSpecification(
	parameters=[
		DistributionModel(
			name="alpha",
			distribution_name="normal",
			distribution_args=[0.5, 0.02],
			data_type=ParameterDataType.CONTINUOUS,
		),
		DistributionModel(
			name="beta",
			distribution_name="normal",
			distribution_args=[0.024, 0.001],
			data_type=ParameterDataType.CONTINUOUS,
		),
	]
)


def history_matching_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


outdir = get_examples_outdir()
specification = HistoryMatchingMethodModel(
	experiment_name="pyesmda_history_matching",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	method="esmda_rs",
	n_samples=50,
	n_iterations=20,
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	covariance=np.eye(observed_data.lynx.values.shape[0]),
	calibration_func_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(save_ensembles_history=True),
	n_jobs=10,
)

calibrator = HistoryMatchingMethod(
	calibration_func=history_matching_func,
	specification=specification,
	engine="pyesmda",
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")
