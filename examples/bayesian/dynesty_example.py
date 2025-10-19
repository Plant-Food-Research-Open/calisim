import numpy as np
import pandas as pd

from calisim.bayesian import (
	BayesianCalibrationMethod,
	BayesianCalibrationMethodModel,
)
from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.statistics import GaussianLogLikelihood
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


def bayesian_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)
	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	metric = GaussianLogLikelihood()

	ll = metric.calculate(observed_data, simulated_data)
	return ll


outdir = get_examples_outdir()
specification = BayesianCalibrationMethodModel(
	experiment_name="dynesty_bayesian_calibration",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	method="dynamic",
	n_init=500,
	n_samples=50000,
	n_iterations=50000,
	n_jobs=1,
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	calibration_func_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(dlogz_init=0.1, nlive_init=200, nlive_batch=100),
)

calibrator = BayesianCalibrationMethod(
	calibration_func=bayesian_func, specification=specification, engine="dynesty"
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")
