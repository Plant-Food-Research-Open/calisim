"""Contains utilities for dispatching jobs using TorchX.

This module defines various utility functions for dispatching
batch jobs using the TorchX launcher.

"""

import os
from typing import Any

import torchx.specs as specs
from torchx.runner import get_runner
from torchx.runner.config import apply

from ..data_model import OrchestrationModel


def get_def(orchestration: OrchestrationModel, args: list[str]) -> specs.AppDef:
	"""Get the application definition.

	Args:
		orchestration (OrchestrationModel): The orchestration data model.
		args (list[str]): The script arguments.

	Returns:
		specs.AppDef: The application definition.
	"""
	entrypoint = orchestration.entrypoint

	args = [str(arg) for arg in args]
	# if entrypoint == "sh" or entrypoint == "/bin/bash" and args[0] != "-c":
	# 	args = ["-c"] + args

	return specs.AppDef(
		name=orchestration.name,
		roles=[
			specs.Role(
				name=f"{orchestration.name}-worker",
				entrypoint=entrypoint,
				args=args,
				image=orchestration.image,
				resource=specs.Resource(
					**dict(
						cpu=orchestration.cpu,
						gpu=orchestration.gpu,
						memMB=orchestration.memMB,
					)
				),
				num_replicas=orchestration.num_replicas,
			)
		],
	)


def launch(
	orchestration: OrchestrationModel, definition: specs.AppDef
) -> dict[str, Any]:
	"""Dispatch a TorchX job using a scheduler.

	Args:
		orchestration (OrchestrationModel): The orchestration data model.
		definition (specs.AppDef): The application definition.
	"""
	scheduler = orchestration.scheduler
	cfg = {
		"partition": orchestration.partition,
		"time": orchestration.time,
		"auto_set_cuda_visible_devices": orchestration.auto_set_cuda_visible_devices,
		"log_dir": orchestration.log_dir,
	}

	apply(scheduler, cfg, dirs=[os.getenv("PWD", "~")])
	runner = get_runner()
	try:
		app_handle = runner.run(definition, scheduler=scheduler, cfg=cfg)
		status = runner.wait(app_handle, wait_interval=orchestration.wait_interval)

		role_name = f"{orchestration.name}-worker"
		logs = runner.log_lines(app_handle, role_name=role_name)

		return {"state": status.state, "logs": logs, "handle": app_handle}
	finally:
		runner.close()


class TorchXJobLauncher:
	"""Utility wrapper around TorchX job orchestration."""

	def launch(
		self, orchestration: OrchestrationModel, args: list[str]
	) -> dict[str, Any]:
		"""Launch the job.

		Args:
		    orchestration (OrchestrationModel): The orchestration data model.
		    args (list[str]): The script arguments.
		"""
		definition = get_def(orchestration, args)
		return launch(orchestration, definition)
