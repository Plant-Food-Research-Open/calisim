"""Exercise hook protocol, path boundaries, and checks without scientific imports."""

import io
import json
import subprocess  # nosec B404 - Used only for result and exception fixtures.
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.agents import hook


class HookTests(unittest.TestCase):
	"""Check observable hook behavior using isolated temporary files."""

	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory()
		self.addCleanup(self.temporary.cleanup)
		self.root = Path(self.temporary.name).resolve()
		self.python = self.root / "example.py"
		self.python.write_text("value = 1\n")

	def test_claude_absolute_path(self) -> None:
		payload = {"tool_input": {"file_path": str(self.python)}}
		self.assertEqual(hook.edited_paths(payload, self.root), [Path("example.py")])

	def test_patch_updates_additions_and_moves(self) -> None:
		(self.root / "new.py").write_text("value = 2\n")
		payload = {
			"tool_name": "apply_patch",
			"tool_input": {
				"command": "*** Begin Patch\n*** Update File: example.py\n"
				"*** Delete File: deleted.py\n*** Move to: new.py\n*** End Patch"
			},
		}
		self.assertEqual(
			hook.edited_paths(payload, self.root),
			[Path("example.py"), Path("new.py")],
		)

	def test_freeform_patch(self) -> None:
		for newline in ("\n", "\r\n"):
			with self.subTest(newline=repr(newline)):
				payload = {
					"tool_name": "apply_patch",
					"tool_input": "*** Add File: example.py\n+value = 1\n".replace(
						"\n", newline
					),
				}
				self.assertEqual(
					hook.edited_paths(payload, self.root), [Path("example.py")]
				)

	def test_relative_path_uses_event_working_directory(self) -> None:
		nested = self.root / "nested"
		nested.mkdir()
		(nested / "example.py").write_text("value = 2\n")
		payload = {"cwd": str(nested), "tool_input": {"file_path": "example.py"}}
		self.assertEqual(
			hook.edited_paths(payload, self.root), [Path("nested/example.py")]
		)

	def test_symlink_escape_is_not_checked(self) -> None:
		with tempfile.TemporaryDirectory() as outside:
			external = Path(outside) / "outside.py"
			external.write_text("value = 2\n")
			link = self.root / "link.py"
			link.symlink_to(external)
			self.assertEqual(
				hook.edited_paths({"tool_input": {"file_path": str(link)}}, self.root),
				[],
			)

	def test_irrelevant_missing_and_environment_paths_are_ignored(self) -> None:
		(self.root / ".venv").mkdir()
		(self.root / ".venv/inside.py").write_text("value = 3\n")
		(self.root / "note.md").write_text("Hello\n")
		for name in ("note.md", "missing.py", "../external.py", ".venv/inside.py"):
			with self.subTest(path=name):
				self.assertEqual(
					hook.edited_paths({"tool_input": {"file_path": name}}, self.root),
					[],
				)

	def test_checks_do_not_rewrite_or_interpret_file_names(self) -> None:
		name = Path("odd name;echo injected.py")
		before = self.python.read_bytes()
		result = subprocess.CompletedProcess([], 1, "lint problem", "")
		with (
			patch.object(hook, "ruff_command", return_value=["ruff"]),
			patch.object(hook.subprocess, "run", return_value=result) as run,
		):
			feedback = hook.check_paths([name], self.root)
		self.assertIn("lint problem", feedback)
		self.assertEqual(run.call_count, 2)
		for call in run.call_args_list:
			args = call.args[0]
			self.assertEqual(args[-2:], ["--", str(name)])
			self.assertNotIn("--fix", args)
			self.assertNotIn("shell", call.kwargs)
		self.assertEqual(self.python.read_bytes(), before)

	def test_real_ruff_does_not_apply_configured_fixes(self) -> None:
		command = hook.ruff_command(hook.ROOT)
		if command is None:
			self.skipTest("Ruff is not installed; unit tests remain available")
		self.python.write_text("import os\nvalue= 1\n")
		before = self.python.read_bytes()
		for settings in (
			"fix = true\n",
			"fix-only = true\n",
			"fix = true\nfix-only = true\n",
		):
			with self.subTest(settings=settings):
				(self.root / "pyproject.toml").write_text("[tool.ruff]\n" + settings)
				with patch.object(hook, "ruff_command", return_value=command):
					feedback = hook.check_paths([Path("example.py")], self.root)
				self.assertIn("F401", feedback)
				self.assertEqual(self.python.read_bytes(), before)

	def test_missing_ruff_reports_incomplete_validation(self) -> None:
		with patch.object(hook, "ruff_command", return_value=None):
			self.assertIn(
				"not checked", hook.check_paths([Path("example.py")], self.root)
			)

	def test_timeout_reports_incomplete_validation(self) -> None:
		with (
			patch.object(hook, "ruff_command", return_value=["ruff"]),
			patch.object(
				hook.subprocess,
				"run",
				side_effect=subprocess.TimeoutExpired("ruff", 15),
			),
		):
			self.assertIn(
				"could not complete", hook.check_paths([Path("example.py")], self.root)
			)

	def test_session_emits_client_compatible_context(self) -> None:
		output = io.StringIO()
		with (
			patch.object(
				hook.sys, "stdin", io.StringIO('{"hook_event_name":"SessionStart"}')
			),
			redirect_stdout(output),
		):
			hook.main()
		response = json.loads(output.getvalue())["hookSpecificOutput"]
		self.assertEqual(response["hookEventName"], "SessionStart")
		self.assertIn("AGENTS.md", response["additionalContext"])

	def test_edit_feedback_uses_post_tool_protocol_without_blocking(self) -> None:
		payload = {
			"hook_event_name": "PostToolUse",
			"tool_name": "Edit",
			"tool_input": {"file_path": str(self.python)},
		}
		output = io.StringIO()
		with (
			patch.object(hook, "ROOT", self.root),
			patch.object(hook.sys, "stdin", io.StringIO(json.dumps(payload))),
			patch.object(hook, "ruff_command", return_value=None),
			redirect_stdout(output),
		):
			hook.main()
		response = json.loads(output.getvalue())
		self.assertEqual(response["hookSpecificOutput"]["hookEventName"], "PostToolUse")
		self.assertIn(
			"not checked", response["hookSpecificOutput"]["additionalContext"]
		)
		self.assertNotIn("decision", response)

	def test_unrelated_and_malformed_events_do_not_run_commands(self) -> None:
		payloads: list[object] = [
			[],
			{"hook_event_name": "Other"},
			{"hook_event_name": "PostToolUse", "tool_name": "Bash"},
			{"hook_event_name": "PostToolUse", "tool_name": []},
			{"hook_event_name": "PostToolUse", "tool_name": "Edit", "tool_input": []},
			{"hook_event_name": "PostToolUse", "tool_name": "Edit", "cwd": []},
		]
		for payload in payloads:
			with (
				self.subTest(payload=payload),
				patch.object(hook.sys, "stdin", io.StringIO(json.dumps(payload))),
				patch.object(hook.subprocess, "run") as run,
			):
				hook.main()
				run.assert_not_called()


if __name__ == "__main__":
	unittest.main()
