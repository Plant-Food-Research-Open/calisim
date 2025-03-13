import numpy as np
import pandas as pd

from calisim.active_learning import (
	ActiveLearningMethod,
	ActiveLearningMethodModel,
)
from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
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


def active_learning_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


outdir = get_examples_outdir()
specification = ActiveLearningMethodModel(
	experiment_name="skactiveml_active_learning",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	output_labels=["Lynx"],
	n_init=20,
	n_iterations=10,
	n_samples=50,
	method="gp",
	query_strategy="greedy_sampling_target",
	method_kwargs=dict(alpha=1e-10, optimizer="fmin_l_bfgs_b"),
	calibration_func_kwargs=dict(t=observed_data.year),
	batched=False,
)


calibrator = ActiveLearningMethod(
	calibration_func=active_learning_func,
	specification=specification,
	engine="skactiveml",
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")
