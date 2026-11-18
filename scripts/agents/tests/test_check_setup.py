"""Exercise setup failures and protected-directory overlay resolution."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
	"check_setup", Path(__file__).resolve().parents[1] / "check_setup.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SetupValidator = MODULE.SetupValidator


class SetupValidatorTests(unittest.TestCase):
	"""Build tiny real configurations so checks cover behavior rather than wording."""

	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory()
		self.addCleanup(self.temporary.cleanup)
		self.root = Path(self.temporary.name) / "repo"
		self.overlay = Path(self.temporary.name) / "prepared"
		self.write(self.root, "AGENTS.md", "[Skills](.agents/skills/check/SKILL.md)\n")
		self.write(self.root, "CLAUDE.md", "[Instructions](AGENTS.md)\n")
		metadata = "---\nname: check\ndescription: Check the change.\n---\n"
		self.write(
			self.overlay,
			".agents/skills/check/SKILL.md",
			metadata + "[Guide](../../../AGENTS.md)\n",
		)
		self.write(self.root, ".claude/skills/check/SKILL.md", metadata)
		self.write(
			self.root,
			".claude/agents/reviewer.md",
			"---\nname: reviewer\ndescription: Review.\n---\n",
		)
		self.write(
			self.overlay,
			".codex/agents/reviewer.toml",
			'name = "reviewer"\ndescription = "Review."\n'
			'developer_instructions = "Read sources."\n',
		)
		self.write(
			self.overlay,
			".codex/config.toml",
			'[agents.reviewer]\ndescription = "Review."\n'
			'config_file = "agents/reviewer.toml"\n',
		)
		self.write(self.root, "scripts/agents/hook.py", "pass\n")
		hook = {
			"type": "command",
			"command": "python3 scripts/agents/hook.py",
			"timeout": 10,
		}
		self.settings = {
			"hooks": {
				"SessionStart": [{"hooks": [hook]}],
				"PostToolUse": [{"matcher": "Write|Edit", "hooks": [hook]}],
			}
		}
		self.write(self.overlay, ".codex/hooks.json", json.dumps(self.settings))
		self.write(self.root, ".claude/settings.json", json.dumps(self.settings))

	@staticmethod
	def write(root: Path, path: str, content: str) -> None:
		"""Write a fixture file and create its parents."""
		target = root / path
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(content, encoding="utf-8")

	def validate(self) -> Any:
		"""Check the temporary project using its separate configuration overlay."""
		validator = SetupValidator(self.root, self.overlay)
		validator.check()
		return validator

	def test_overlay_resolves_configuration_and_repository_links(self) -> None:
		validator = self.validate()
		self.assertEqual(validator.errors, [])
		self.assertEqual(validator.counts["skills"], 1)
		self.assertEqual(validator.counts["agents"], 1)
		self.assertEqual(validator.counts["links"], 3)

	def test_overlay_does_not_replace_claude_files(self) -> None:
		self.write(self.overlay, ".claude/settings.json", "invalid")
		self.assertEqual(self.validate().errors, [])

	def test_local_settings_transcripts_and_nested_worktrees_are_not_read(self) -> None:
		for base, path in (
			(self.root, ".claude/settings.local.json"),
			(self.root, ".claude/transcripts/session.json"),
			(self.root, ".claude/worktrees/other/.claude/settings.json"),
			(self.overlay, ".codex/local.json"),
			(self.overlay, ".codex/worktrees/other/.codex/config.toml"),
		):
			self.write(base, path, "malformed private configuration")
		validator = self.validate()
		self.assertEqual(validator.errors, [])
		self.assertEqual(validator.counts["configs"], 4)

	def test_each_client_requires_both_shared_hook_events(self) -> None:
		for base, path in (
			(self.root, ".claude/settings.json"),
			(self.overlay, ".codex/hooks.json"),
		):
			for missing in ("SessionStart", "PostToolUse"):
				with self.subTest(path=path, event=missing):
					settings = json.loads(json.dumps(self.settings))
					del settings["hooks"][missing]
					self.write(base, path, json.dumps(settings))
					self.assertTrue(
						any(missing in error for error in self.validate().errors)
					)
			self.write(base, path, json.dumps(self.settings))

	def test_unrelated_events_and_handlers_do_not_need_our_entrypoint(self) -> None:
		settings = json.loads(json.dumps(self.settings))
		extra = {"type": "command", "command": "printf done", "timeout": 5}
		settings["hooks"]["Stop"] = [{"hooks": [extra]}]
		settings["hooks"]["SessionStart"][0]["hooks"].append(extra)
		settings["hooks"]["PostToolUse"].append({"hooks": [extra]})
		self.write(self.root, ".claude/settings.json", json.dumps(settings))
		validator = self.validate()
		self.assertEqual(validator.errors, [])
		self.assertEqual(validator.counts["hooks"], 4)

	def test_malformed_configs_and_missing_registration_are_reported(self) -> None:
		self.write(self.overlay, ".codex/config.toml", "[invalid")
		self.write(self.root, ".claude/settings.json", "{")
		validator = self.validate()
		self.assertTrue(any("cannot parse" in error for error in validator.errors))
		self.assertTrue(any("agents.reviewer" in error for error in validator.errors))

	def test_mismatched_metadata_and_missing_wrapper_are_reported(self) -> None:
		self.write(
			self.overlay,
			".agents/skills/check/SKILL.md",
			"---\nname: wrong\ndescription: ''\n---\n",
		)
		(self.root / ".claude/skills/check/SKILL.md").unlink()
		validator = self.validate()
		self.assertTrue(any("name must match" in error for error in validator.errors))
		self.assertTrue(
			any("description must be" in error for error in validator.errors)
		)
		self.assertTrue(
			any(".claude/skills/check/SKILL.md" in error for error in validator.errors)
		)

	def test_unsafe_or_missing_role_target_is_rejected(self) -> None:
		for target in (
			str(self.root.parent / "outside.toml"),
			"../outside.toml",
			"agents/missing.toml",
		):
			with self.subTest(target=target):
				self.write(
					self.overlay,
					".codex/config.toml",
					'[agents.reviewer]\ndescription = "Review."\n'
					f'config_file = "{target}"\n',
				)
				self.assertTrue(
					any("config_file" in error for error in self.validate().errors)
				)

	def test_hooks_require_command_type_script_and_positive_timeout(self) -> None:
		for hook in (
			{"type": "prompt", "command": "scripts/agents/hook.py", "timeout": 10},
			{"type": "command", "command": "missing.py", "timeout": 10},
			{"type": "command", "command": "scripts/agents/hook.py", "timeout": 0},
		):
			with self.subTest(hook=hook):
				self.write(
					self.root,
					".claude/settings.json",
					json.dumps(
						{
							"hooks": {
								"SessionStart": self.settings["hooks"]["SessionStart"],
								"PostToolUse": [
									{"matcher": "Write|Edit", "hooks": [hook]}
								],
							}
						}
					),
				)
				self.assertTrue(self.validate().errors)

	def test_links_skip_urls_and_anchors_but_report_missing_files(self) -> None:
		self.write(
			self.root,
			"AGENTS.md",
			"[Web](https://example.com) [Anchor](#rules) [Missing](docs/missing.md)\n",
		)
		validator = self.validate()
		self.assertEqual(len(validator.errors), 1)
		self.assertIn("docs/missing.md", validator.errors[0])


if __name__ == "__main__":
	unittest.main()
