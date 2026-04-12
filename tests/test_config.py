"""
Tests for various configuration utilities and functions.

A battery of tests to validate configuration utilities and functions.

"""

import os.path as osp

from omegaconf import DictConfig

from calisim.base import ExampleModelContainer
from calisim.config import HydraConfiguration
from calisim.statistics import RootMeanSquaredError

from .conftest import get_calibration_func, is_close


def get_hydra_config(
	overrides: list[str] | None = None,
) -> tuple[DictConfig, HydraConfiguration]:
	"""Get the Hydra configuration.

	Args:
	    overrides (list[str] | None, optional): Overrides for the
			configuration. Defaults to None.

	Returns:
	    tuple[DictConfig, HydraConfiguration]: The loaded Hydra configuration
	    and the Hydra configuration helper instance.
	"""
	hydra_config = HydraConfiguration()
	cfg = hydra_config.get_configuration(
		"config", osp.join("tests", "conf"), overrides=overrides
	)
	return cfg, hydra_config


def test_hydra_dict() -> None:
	cfg, hydra_config = get_hydra_config()
	cfg_dict = hydra_config.to_dict(cfg)
	assert isinstance(cfg_dict, dict)


def test_hydra_merge() -> None:
	first_cfg, _ = get_hydra_config()
	second_cfg, hydra_config = get_hydra_config(
		overrides=["metric._target_=calisim.statistics.RootMeanSquaredError"]
	)

	merged_cfg = hydra_config.merge(first_cfg, second_cfg)
	assert merged_cfg is not None
	assert isinstance(merged_cfg["metric"], RootMeanSquaredError)


def test_hydra_overrides() -> None:
	cfg, hydra_config = get_hydra_config()
	merged_cfg = hydra_config.apply_overrides(
		cfg, overrides=["metric._target_=calisim.statistics.RootMeanSquaredError"]
	)

	assert merged_cfg is not None
	metric_instance = hydra_config.instantiate(merged_cfg["metric"])
	assert isinstance(metric_instance, RootMeanSquaredError)


def test_hydra_pretty() -> None:
	cfg, hydra_config = get_hydra_config()
	pretty_cfg = hydra_config.pretty(cfg)
	assert pretty_cfg != ""


def test_hydra_valid() -> None:
	cfg, hydra_config = get_hydra_config()
	assert hydra_config.validate(cfg)


def test_hydra_available() -> None:
	_, hydra_config = get_hydra_config()
	assert hydra_config.available


def test_hydra_optimisation() -> None:
	cfg, _ = get_hydra_config()
	metric = cfg["metric"]
	model = cfg["model"]
	observed_data = model.get_observed_data()
	model_container = ExampleModelContainer(model)
	output_label = model_container.output_labels[0]

	calibration_func = get_calibration_func(model_container, output_label, metric)

	calibrator = cfg["calibration"](calibration_func=calibration_func)
	calibrator.specification.calibration_func_kwargs = dict(t=observed_data.year)
	calibrator.specification.observed_data = observed_data[output_label].values
	calibrator.specify().execute().analyze()
	assert is_close(model_container, calibrator)
