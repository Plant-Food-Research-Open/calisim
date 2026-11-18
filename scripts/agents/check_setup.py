#!/usr/bin/env python3
"""Validate repository agent files without installing project dependencies."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import tomllib


class SetupValidator:
	"""Check structural contracts and links using an optional configuration overlay."""

	def __init__(self, root: Path, config_root: Path | None = None) -> None:
		self.root = root.resolve()
		self.config_root = config_root.resolve() if config_root else self.root
		self.errors: list[str] = []
		self.counts = dict(configs=0, skills=0, agents=0, hooks=0, links=0)
		self.configs: dict[Path, Any] = {}

	def path(self, relative: Path | str) -> Path:
		"""Resolve repository paths, overlaying only protected agent folders."""
		relative = Path(relative)
		base = (
			self.config_root
			if relative.parts[0] in {".agents", ".codex"}
			else self.root
		)
		return base / relative

	def error(self, path: Path | str, message: str) -> None:
		"""Record an actionable error identified by repository-relative path."""
		self.errors.append(f"{path}: {message}")

	def metadata(self, path: Path, expected_name: str) -> None:
		"""Check our flat name/description fields, rather than parsing general YAML."""
		try:
			text = self.path(path).read_text(encoding="utf-8")
		except (OSError, UnicodeError) as error:
			self.error(path, str(error))
			return
		frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.S)
		if not frontmatter:
			self.error(
				path, "missing YAML frontmatter with flat name/description fields"
			)
			return
		fields = dict(
			re.findall(r"^(name|description):[ \t]*(.*)$", frontmatter[1], re.M)
		)
		for field in ("name", "description"):
			value = fields.get(field, "").strip()
			if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
				value = value[1:-1]
			if not value or value in {">", "|", ">-", "|-"} or value.startswith("#"):
				self.error(path, f"{field} must be a nonempty, single-line string")
			elif field == "name" and value != expected_name:
				self.error(path, f"name must match {expected_name!r}, found {value!r}")

	def parse_configs(self) -> None:
		"""Parse shared entrypoints without inspecting local settings or transcripts."""
		paths = {
			Path(".codex/config.toml"),
			Path(".codex/hooks.json"),
			Path(".claude/settings.json"),
		}
		paths.update(
			Path(".codex/agents") / file.name
			for file in self.path(".codex/agents").glob("*.toml")
		)
		for logical in sorted(paths):
			try:
				text = self.path(logical).read_text(encoding="utf-8")
				self.configs[logical] = (
					json.loads(text)
					if logical.suffix == ".json"
					else tomllib.loads(text)
				)
				self.counts["configs"] += 1
			except (OSError, ValueError) as error:
				self.error(logical, f"cannot parse configuration: {error}")

	def config(self, path: str) -> dict[str, Any]:
		"""Return a parsed configuration object, diagnosing missing files or types."""
		data = self.configs.get(Path(path))
		if not isinstance(data, dict):
			self.error(path, "expected a readable JSON/TOML object")
			return {}
		return data

	def check_skills(self) -> None:
		"""Check canonical skill names, descriptions, and Claude discovery wrappers."""
		base = self.path(".agents/skills")
		if not base.is_dir():
			self.error(".agents/skills", "canonical skill directory is missing")
		for directory in sorted(base.iterdir()) if base.is_dir() else []:
			if not directory.is_dir():
				continue
			name = directory.name
			self.metadata(Path(".agents/skills") / name / "SKILL.md", name)
			self.metadata(Path(".claude/skills") / name / "SKILL.md", name)
			self.counts["skills"] += 1

	def check_agents(self) -> None:
		"""Check agent definitions, matching Claude files, and Codex registrations."""
		registrations = self.config(".codex/config.toml").get("agents", {})
		if not isinstance(registrations, dict):
			self.error(".codex/config.toml", "agents must be a table")
			registrations = {}
		for file in sorted(self.path(".codex/agents").glob("*.toml")):
			logical = Path(".codex/agents") / file.name
			data = self.config(str(logical))
			for field in ("name", "description", "developer_instructions"):
				if not isinstance(data.get(field), str) or not data[field].strip():
					self.error(logical, f"{field} must be a nonempty string")
			name = file.stem
			if data.get("name") != name:
				self.error(logical, f"name must match filename {name!r}")
			self.metadata(Path(".claude/agents") / f"{name}.md", name)
			registration = registrations.get(name, {})
			if not isinstance(registration, dict) or not registration.get(
				"description"
			):
				self.error(".codex/config.toml", f"missing agents.{name} description")
			if (
				not isinstance(registration, dict)
				or registration.get("config_file") != f"agents/{file.name}"
			):
				self.error(
					".codex/config.toml",
					f"agents.{name}.config_file must target agents/{file.name}",
				)
			self.counts["agents"] += 1
		for name, registration in registrations.items():
			if not isinstance(registration, dict):
				continue  # Numeric concurrency settings may also appear in this table.
			target = registration.get("config_file")
			if (
				not isinstance(target, str)
				or Path(target).is_absolute()
				or ".." in Path(target).parts
			):
				self.error(
					".codex/config.toml",
					f"agents.{name}.config_file must be a relative path",
				)
			elif not self.path(Path(".codex") / target).is_file():
				self.error(
					".codex/config.toml",
					f"agents.{name}.config_file target does not exist: {target}",
				)

	def check_hooks(self) -> None:
		"""Check known lifecycle registrations and their shared hook entrypoint."""
		for path in (".codex/hooks.json", ".claude/settings.json"):
			events = self.config(path).get("hooks")
			if not isinstance(events, dict) or not events:
				self.error(path, "hooks must be a nonempty object")
				continue
			for event in ("SessionStart", "PostToolUse"):
				groups = events.get(event, [])
				if not isinstance(groups, list):
					self.error(path, f"{event} must contain a list of hook groups")
					continue
				found = False
				for group in groups:
					hooks = group.get("hooks", []) if isinstance(group, dict) else []
					if not isinstance(hooks, list):
						continue
					for hook in hooks:
						if not isinstance(hook, dict):
							continue
						command, args = hook.get("command", ""), hook.get("args", [])
						if "scripts/agents/hook.py" not in str(command) + str(args):
							continue
						found = True
						if (
							hook.get("type") != "command"
							or not isinstance(command, str)
							or not command.strip()
							or not isinstance(args, list)
							or not all(isinstance(arg, str) for arg in args)
						):
							self.error(
								path,
								f"{event} must invoke scripts/agents/hook.py "
								"as a command hook",
							)
						timeout = hook.get("timeout")
						if (
							isinstance(timeout, bool)
							or not isinstance(timeout, int | float)
							or not timeout > 0
						):
							self.error(path, f"{event} hook timeout must be positive")
						self.counts["hooks"] += 1
				if not found:
					self.error(path, f"{event} must register scripts/agents/hook.py")
		if not self.path("scripts/agents/hook.py").is_file():
			self.error("scripts/agents/hook.py", "registered hook script is missing")

	def check_links(self) -> None:
		"""Resolve inline Markdown file links against the logical repository tree."""
		paths = {Path("AGENTS.md"), Path("CLAUDE.md")}
		for folder in ("calisim", "tests", "docs", "examples"):
			paths.update(
				path.relative_to(self.root)
				for path in (self.root / folder).rglob("AGENTS.md")
			)
		paths.update(
			path.relative_to(self.root)
			for path in (self.root / "docs/development").glob("*.md")
		)
		for folder in (
			".agents/skills",
			".claude/rules",
			".claude/skills",
			".claude/agents",
		):
			base = self.path(folder)
			paths.update(
				Path(folder) / path.relative_to(base) for path in base.rglob("*.md")
			)
		for path in sorted(paths):
			if not self.path(path).is_file():
				self.error(path, "agent guidance file is missing")
				continue
			for target in re.findall(
				r"\[[^\]]*\]\(([^)]+)\)", self.path(path).read_text(encoding="utf-8")
			):
				target = target.strip().removeprefix("<").removesuffix(">")
				parsed = urlsplit(target)
				if parsed.scheme or parsed.netloc or not parsed.path:
					continue
				resolved = (self.root / path.parent / unquote(parsed.path)).resolve()
				try:
					logical = resolved.relative_to(self.root)
				except ValueError:
					self.error(path, f"local link escapes the repository: {target}")
					continue
				if not self.path(logical).exists():
					self.error(path, f"local link target does not exist: {target}")
				self.counts["links"] += 1

	def check(self) -> bool:
		"""Run independent checks and return whether the setup is valid."""
		self.parse_configs()
		self.check_skills()
		self.check_agents()
		self.check_hooks()
		self.check_links()
		return not self.errors


def main() -> int:
	"""Validate an installed setup or protected configuration prepared elsewhere."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--root", type=Path, default=Path(__file__).resolve().parents[2]
	)
	parser.add_argument(
		"--config-root", type=Path, help="overlay only .agents and .codex"
	)
	args = parser.parse_args()
	validator = SetupValidator(args.root, args.config_root)
	if not validator.check():
		print("Agent setup validation failed:", file=sys.stderr)
		for error in validator.errors:
			print(f"- {error}", file=sys.stderr)
		return 1
	counts = ", ".join(f"{count} {name}" for name, count in validator.counts.items())
	print(f"Agent setup valid: {counts}.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
