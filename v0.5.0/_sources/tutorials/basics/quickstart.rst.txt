Quickstart
==========

The following code demonstrates how one may optimise the `alpha` parameter of the Lotka Volterra model using the `Optuna` black-box optimisation library.

In this example, we minimise the mean squared error between observed and simulated data for the population density of Lynxes.

Finally, we view the results of the optimisation procedure.

.. code-block:: python

    # Load imports
    import numpy as np
    import pandas as pd

    from calisim.data_model import (
        DistributionModel,
        ParameterDataType,
        ParameterSpecification,
    )
    from calisim.example_models import LotkaVolterraModel
    from calisim.optimisation import OptimisationMethod, OptimisationMethodModel
    from calisim.statistics import MeanSquaredError
    from calisim.utils import get_examples_outdir

    # Get model
    model = LotkaVolterraModel()
    observed_data = model.get_observed_data()

    # Specify model parameter distributions
    parameter_spec = ParameterSpecification(
        parameters=[
            DistributionModel(
                name="alpha",
                distribution_name="uniform",
                distribution_args=[0.45, 0.55],
                data_type=ParameterDataType.CONTINUOUS,
            )
        ]
    )

    # Define objective function
    def objective(
        parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
    ) -> float | list[float]:
        simulation_parameters = dict(
            alpha=parameters["alpha"],
            beta=0.024, h0=34.0, l0=5.9,
            t=t, gamma=0.84, delta=0.026,
        )

        simulated_data = model.simulate(simulation_parameters).lynx.values
        metric = MeanSquaredError()
        discrepancy = metric.calculate(observed_data, simulated_data)
        return discrepancy

    # Specify calibration parameter values
    specification = OptimisationMethodModel(
        experiment_name="optuna_optimisation",
        parameter_spec=parameter_spec,
        observed_data=observed_data.lynx.values,
        outdir=get_examples_outdir(),
        method="tpes",
        directions=["minimize"],
        n_iterations=100,
        method_kwargs=dict(n_startup_trials=50),
        calibration_func_kwargs=dict(t=observed_data.year),
    )

    # Choose calibration engine
    calibrator = OptimisationMethod(
        calibration_func=objective, specification=specification, engine="optuna"
    )

    # Run the workflow
    calibrator.specify().execute().analyze()

    # View the results
    result_artifacts = "\n".join(calibrator.get_artifacts())
    print(f"View results: \n{result_artifacts}")
    print(f"Parameter estimates: {calibrator.get_parameter_estimates()}")
