"""Contains utilities for managing Hydra-based calibration configurations.

This module defines utility functions for managing and configuration
calibration workflows via Hydra.

"""

import os

from omegaconf import DictConfig


class HydraConfiguration:
	"""Utility wrapper around Hydra config composition and instantiation."""

	def __init__(self) -> None:
		try:
			from hydra import compose, initialize_config_dir
			from hydra.utils import instantiate

			self._compose = compose
			self._initialize_config_dir = initialize_config_dir
			self._instantiate = instantiate
			self._available = True
		except ImportError:
			self._available = False
			self._compose = None
			self._initialize_config_dir = None
			self._instantiate = None

	def get_raw_config(
		self,
		config_name: str,
		config_dir: str,
		job_name: str = "app",
		overrides: list[str] | None = None,
	) -> dict | None:
		"""Get the raw configuration object.

		Args:
		    config_name (str): The name of the configuration file.
		    config_dir (str): The configuration directory file path
		    job_name (str, optional): The name of the Hydra job. Defaults to "app".
		    overrides (list[str] | None, optional): Overrides for the configuration.
				Defaults to None.

		Returns:
		    dict | None: The configuration object.
		"""
		if not self._available:
			return None

		current_path = os.getcwd()
		full_dir = os.path.join(current_path, config_dir)

		with self._initialize_config_dir(
			version_base=None,
			config_dir=full_dir,
			job_name=job_name,
		):
			return self._compose(
				config_name=config_name,
				overrides=overrides,
			)

	def instantiate(self, cfg: dict) -> DictConfig | None:
		"""Instantiate the raw configuration object.

		Args:
		    cfg (dict): The raw configuration dictionary.

		Returns:
		    DictConfig | None: The instantiated configuration.
		"""
		if not self._available or cfg is None:
			return None
		return self._instantiate(cfg)

	def get_configuration(
		self,
		config_name: str,
		config_dir: str,
		job_name: str = "app",
		overrides: list[str] | None = None,
	) -> DictConfig | None:
		"""Get the configuration object.

		Args:
		    config_name (str): The name of the configuration file.
		    config_dir (str): The configuration directory file path
		    job_name (str, optional): The name of the Hydra job. Defaults to "app".
		    overrides (list[str] | None, optional): Overrides for the configuration.
				Defaults to None.

		Returns:
		    DictConfig | None: The configuration object.
		"""
		cfg = self.get_raw_config(
			config_name=config_name,
			config_dir=config_dir,
			job_name=job_name,
			overrides=overrides,
		)
		if cfg is None:
			return None

		return self.instantiate(cfg)

	def to_dict(self, cfg: DictConfig) -> dict | None:
		"""Convert a configuration object to a dictionary.

		Args:
		    cfg (DictConfig): The configuration object.

		Returns:
		    dict | None: The converted configuration.
		"""
		if not self._available or cfg is None:
			return None

		from omegaconf import OmegaConf

		return OmegaConf.to_container(cfg, resolve=True)

	def merge(self, *configs: DictConfig) -> DictConfig | None:
		"""Merge a list of configuration objects into one.

		Returns:
		    DictConfig | None: The merged configuration object.
		"""
		if not self._available:
			return None

		from omegaconf import OmegaConf

		return OmegaConf.merge(*configs)

	def apply_overrides(
		self,
		cfg: DictConfig,
		overrides: list[str],
	) -> DictConfig:
		"""Override the contents of the configuration.

		Args:
		    cfg (DictConfig): The configuration object.
		    overrides (list[str]): Overrides for the configuration.

		Returns:
		    DictConfig: The overridden configuration.
		"""
		if not self._available or cfg is None:
			return cfg

		from omegaconf import OmegaConf

		override_cfg = OmegaConf.from_dotlist(overrides)
		return OmegaConf.merge(cfg, override_cfg)

	def pretty(self, cfg: DictConfig) -> str:
		"""Prettify the configuration object.

		Args:
		    cfg (DictConfig): The configuration object.

		Returns:
		    str: The configuration object as a string.
		"""
		if not self._available or cfg is None:
			return ""

		from omegaconf import OmegaConf

		return OmegaConf.to_yaml(cfg)

	def validate(self, cfg: DictConfig) -> bool:
		"""Validate the configuration object.

		Args:
		    cfg (DictConfig): The configuration object.

		Returns:
		    bool: Whether the configuration object is valid.
		"""
		if not self._available or cfg is None:
			return False

		try:
			from omegaconf import OmegaConf
			from omegaconf.errors import (
				ConfigAttributeError,
				InterpolationResolutionError,
				MissingMandatoryValue,
				OmegaConfBaseException,
				ValidationError,
			)

			OmegaConf.to_container(cfg, resolve=True)
			return True
		except (
			ValidationError,
			ConfigAttributeError,
			InterpolationResolutionError,
			MissingMandatoryValue,
			OmegaConfBaseException,
		):
			return False

	@property
	def available(self) -> bool:
		"""Check if Hydra is available.

		Returns:
		    bool: Whether Hydra is available.
		"""
		return self._available
