"""
Tests for various orchestration utilities and functions.

A battery of tests to validate orchestration utilities and functions.

"""

from torchx.specs.api import AppState

from calisim.data_model import OrchestrationModel
from calisim.orchestration import TorchXJobLauncher


def test_torchx_python_successful_job() -> None:
	orchestration = OrchestrationModel(
		name="test",
		entrypoint="python",
		cpu=1,
		gpu=0,
		log_dir=None,
		memMB=10,
		scheduler="local_cwd",
		time="00:01:00",
		wait_interval=10,
	)

	runner = TorchXJobLauncher()

	cmd = "1 + 1"
	job_results = runner.launch(orchestration, ["-c", cmd])

	assert job_results["state"] == AppState.SUCCEEDED


def test_torchx_python_failed_job() -> None:
	orchestration = OrchestrationModel(
		name="test",
		entrypoint="python",
		cpu=1,
		gpu=0,
		log_dir=None,
		memMB=10,
		scheduler="local_cwd",
		time="00:01:00",
		wait_interval=10,
	)

	runner = TorchXJobLauncher()

	cmd = "import sys; sys.exit(1)"
	job_results = runner.launch(orchestration, ["-c", cmd])

	assert job_results["state"] == AppState.FAILED


def test_torchx_sh_successful_job() -> None:
	orchestration = OrchestrationModel(
		name="test",
		entrypoint="sh",
		cpu=1,
		gpu=0,
		log_dir=None,
		memMB=10,
		scheduler="local_cwd",
		time="00:01:00",
		wait_interval=10,
	)

	runner = TorchXJobLauncher()

	cmd = "echo $((1+1))"
	job_results = runner.launch(orchestration, ["-c", cmd])

	assert job_results["state"] == AppState.SUCCEEDED


def test_torchx_sh_failed_job() -> None:
	orchestration = OrchestrationModel(
		name="test",
		entrypoint="sh",
		cpu=1,
		gpu=0,
		log_dir=None,
		memMB=10,
		scheduler="local_cwd",
		time="00:01:00",
		wait_interval=10,
	)

	runner = TorchXJobLauncher()

	cmd = "exit 1"
	job_results = runner.launch(orchestration, ["-c", cmd])

	assert job_results["state"] == AppState.FAILED
