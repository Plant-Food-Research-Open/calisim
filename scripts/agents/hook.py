#!/usr/bin/env python3
"""Provide session context and check Python edits for Codex and Claude Code."""

import json
import os
import re
import shutil
import subprocess  # nosec B404 - Ruff runs with an argument list and no shell.
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "apply_patch"}
SESSION_CONTEXT = (
	"calisim: read AGENTS.md and applicable subtree AGENTS.md rules. "
	"Shared skills are in .agents/skills; specialist roles are documented in "
	"docs/development/agent-roles.md. Use calisim-check to choose relevant checks. "
	"Python edit hooks provide Ruff feedback; run scientific tests separately. "
	"Preserve existing contributor changes and report actual validation results."
)


def edited_paths(payload: dict[str, Any], root: Path) -> list[Path]:
	"""Extract existing in-repository Python paths from supported edit payloads."""
	tool_input = payload.get("tool_input", {})
	candidates: list[str] = []
	if isinstance(tool_input, dict):
		file_path = tool_input.get("file_path")
		if isinstance(file_path, str):
			candidates.append(file_path)
		patch = next(
			(
				tool_input[key]
				for key in ("command", "patch", "input")
				if isinstance(tool_input.get(key), str)
			),
			"",
		)
	elif isinstance(tool_input, str):
		patch = tool_input
	else:
		return []
	if payload.get("tool_name") == "apply_patch":
		patch = patch.replace("\r\n", "\n")
		candidates.extend(
			re.findall(
				r"^\*\*\* (?:Add File|Update File|Move to): (.+)$", patch, re.MULTILINE
			)
		)
	base = Path(payload.get("cwd") or root)
	if not base.is_absolute():
		base = root / base
	paths: set[Path] = set()
	for candidate in candidates:
		path = (base / candidate).resolve()
		try:
			relative = path.relative_to(root.resolve())
		except ValueError:
			continue
		if (
			path.suffix in {".py", ".pyi"}
			and path.is_file()
			and not {".git", ".venv", "venv", "__pycache__"}.intersection(
				relative.parts
			)
		):
			paths.add(relative)
	return sorted(paths)


def ruff_command(root: Path) -> list[str] | None:
	"""Use an existing project interpreter or Ruff on PATH without installing it."""
	for relative in (".venv/bin/python", ".venv/Scripts/python.exe"):
		interpreter = root / relative
		if interpreter.is_file() and os.access(interpreter, os.X_OK):
			return [str(interpreter), "-m", "ruff"]
	executable = shutil.which("ruff")
	return [executable] if executable else None


def check_paths(paths: list[Path], root: Path) -> str:
	"""Return failed-check feedback without changing source files or using a shell."""
	if not paths:
		return ""
	command = ruff_command(root)
	if command is None:
		return (
			"Python edits were not checked: Ruff is unavailable. Configure the "
			"project dev environment and run the checks in docs/development/README.md."
		)
	messages = []
	for options in (
		["check", "--no-fix", "--no-fix-only", "--output-format", "concise"],
		["format", "--check"],
	):
		args = [
			*command,
			*options,
			"--no-cache",
			"--force-exclude",
			"--",
			*(str(path) for path in paths),
		]
		try:
			# Arguments are passed directly; file names are never shell commands.
			result = subprocess.run(  # nosec B603
				args,
				cwd=root,
				capture_output=True,
				text=True,
				timeout=15,
				check=False,
			)
		except (OSError, subprocess.TimeoutExpired) as error:
			messages.append(f"Ruff {' '.join(options)} could not complete: {error}")
			break
		if result.returncode:
			output = (result.stdout + result.stderr).strip()
			messages.append(f"Ruff {' '.join(options)} failed:\n{output}")
	return "\n\n".join(messages)[:6000]


def main() -> None:
	"""Read one hook event and emit advisory context using the shared JSON schema."""
	try:
		payload = json.load(sys.stdin)
	except (ValueError, OSError):
		print("calisim hook: ignored invalid JSON input.", file=sys.stderr)
		return
	if not isinstance(payload, dict):
		return
	event = payload.get("hook_event_name")
	tool_name = payload.get("tool_name")
	if event == "SessionStart":
		context = SESSION_CONTEXT
	elif (
		event == "PostToolUse"
		and isinstance(tool_name, str)
		and tool_name in EDIT_TOOLS
	):
		if not isinstance(payload.get("cwd", ""), str):
			return
		context = check_paths(edited_paths(payload, ROOT), ROOT)
	else:
		return
	if context:
		print(
			json.dumps(
				{
					"hookSpecificOutput": {
						"hookEventName": event,
						"additionalContext": context,
					}
				}
			)
		)


if __name__ == "__main__":
	main()
