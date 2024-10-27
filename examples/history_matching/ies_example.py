import numpy as np
import pandas as pd

from calisim.data_model import DistributionModel
from calisim.example_models import LotkaVolterraModel
from calisim.history_matching import (
	HistoryMatchingMethod,
	HistoryMatchingMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	DistributionModel(name="alpha", dist_name="normal", dist_args=[0.4, 0.03]),
	DistributionModel(name="beta", dist_name="normal", dist_args=[0.025, 0.003]),
]


def history_matching_func(
	parameters: dict, _: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


specification = HistoryMatchingMethodModel(
	experiment_name="ies_history_matching",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=get_examples_outdir(),
	method="sies",
	n_samples=50,
	n_iterations=10,
	output_labels=["Lynx"],
	verbose=True,
	vectorize=False,
	covariance=np.eye(observed_data.lynx.values.shape[0]),
	calibration_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(truncation=1.0),
)

calibrator = HistoryMatchingMethod(
	calibration_func=history_matching_func, specification=specification, engine="ies"
)

calibrator.specify().execute().analyze()
