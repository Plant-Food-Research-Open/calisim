import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.likelihood_free import (
	LikelihoodFreeMethod,
	LikelihoodFreeMethodModel,
)
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = ParameterSpecification(
	parameters=[
		DistributionModel(
			name="alpha",
			distribution_name="normal",
			distribution_args=[0.51, 0.02],
			distribution_bounds=[0.3, 0.7],
			data_type=ParameterDataType.CONTINUOUS,
		),
		DistributionModel(
			name="beta",
			distribution_name="normal",
			distribution_args=[0.024, 0.001],
			distribution_bounds=[0.01, 0.04],
			data_type=ParameterDataType.CONTINUOUS,
		),
	]
)


def likelihood_free_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy


outdir = get_examples_outdir()
specification = LikelihoodFreeMethodModel(
	experiment_name="elfi_likelihood_free",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_init=25,
	n_samples=100,
	walltime=3,  # minutes
	n_iterations=100,
	n_chains=4,
	acq_noise_var=0,
	method="bolfi",
	sampler="metropolis",  # or nuts
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	calibration_func_kwargs=dict(t=observed_data.year),
)

calibrator = LikelihoodFreeMethod(
	calibration_func=likelihood_free_func, specification=specification, engine="elfi"
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")
