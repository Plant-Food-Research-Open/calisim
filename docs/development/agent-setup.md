# Agent setup and maintenance

The repository has shared instructions and workflows plus small native entry
files for Codex and Claude Code. All commands below run from the repository root
with Python 3.11 or 3.12.

| Capability | Shared source | Codex | Claude Code |
| --- | --- | --- | --- |
| Repository guidance | `AGENTS.md` and subtree `AGENTS.md` files | Native discovery | `CLAUDE.md` plus scoped rule pointers |
| Skills | `.agents/skills/<name>/SKILL.md` | Native discovery | `.claude/skills/<name>/SKILL.md` loaders |
| Specialist roles | [agent-roles.md](agent-roles.md) | `.codex/agents/*.toml`, registered in `.codex/config.toml` | `.claude/agents/*.md` |
| Lifecycle hooks | [hook.py](../../scripts/agents/hook.py) | `.codex/hooks.json` | `.claude/settings.json` |
| Configuration checks | [check_setup.py](../../scripts/agents/check_setup.py) | Local command, pre-commit, and CI | Same |

Claude skill loaders read the corresponding canonical skill before doing its
work. They are ordinary files, so checkout does not depend on symlink support.
Copilot continues to use `.github/copilot-instructions.md` and can follow the
shared files when instructed; it does not gain these hooks or roles automatically.

## Available skills

- `calisim-backend`: add or change a scientific backend, its task specification,
  or workflow dispatch.
- `calisim-check`: choose and execute relevant checks, diagnose failures, and
  report their actual scope.
- `calisim-scientific-review`: review numerical and scientific correctness
  without changing the implementation.
- `calisim-docs`: update examples, API references, tutorials, or contributor
  documentation and verify the relevant outputs.

For example, ask “Use $calisim-backend to add this engine” in Codex or
“/calisim-check validate this change” in Claude Code. Skills remain eligible for
automatic selection; invoking a skill does not authorize publishing or broaden
the task.

## Subagents and rules

[agent-roles.md](agent-roles.md) defines `code-explorer`,
`backend-implementer`, and `scientific-reviewer`. Start a fresh session after
adding agent definitions and use the client's agent selector or ask it to use a
role. Hosts that do not expose repository-defined agents can still use those
role instructions with an available delegate.

Subtree `AGENTS.md` files provide rules for library code, tests, examples, and
documentation. Claude's path-scoped rules load pointers to those same files.
Codex execution-policy `.rules` files govern shell escalation, a separate concern;
this setup leaves the session's existing permission policy in control.

## Hooks

The same standard-library script serves both clients:

- `SessionStart` supplies a short reminder of project instructions and workflows,
  including after context compaction.
- `PostToolUse` runs Ruff lint and format checks on existing Python files named by
  an `Edit`, `Write`, `MultiEdit`, or `apply_patch` event. It handles the Codex
  patch text and Claude's `file_path` payload.

Checks use the project's existing virtual environment, or Ruff already on PATH.
They never install dependencies or rewrite source files. Missing tools, timeouts,
and failed checks return concise feedback to the assistant. Files outside the
repository and symlink escapes are ignored. Edits through a shell command are
covered by explicit validation and pre-commit, rather than this edit hook.

These hooks provide feedback after an edit; they cannot undo it or establish a
security boundary. They do not run the scientific test suite after every edit.
They are POSIX shell configurations, suitable for Linux, macOS, and WSL. A native
Windows client needs equivalent interpreter commands in its hook configuration.

Review new hooks in the client's normal trust interface before activating them.
In Codex, use `/hooks`; project hooks also require a trusted project. In Claude
Code, inspect hooks with `/hooks` and use its workspace trust flow. This repository
does not preapprove hooks or change sandbox, model, or approval settings.

## Validate the setup

```bash
python3 scripts/agents/check_setup.py
python3 -m unittest discover -s scripts/agents/tests -v
poetry run ruff check scripts/agents
poetry run ruff format --check scripts/agents
```

The structural checker verifies configuration, required skill fields, role
references, and local links. It does not start a model session or prove a client
activated a hook. Unit tests exercise hook payloads and file boundaries without
loading the scientific dependencies. The existing pre-commit configuration also
runs the structural checker, and `agent-tooling.yaml` runs these checks in CI.

To smoke-test the feedback manually, without invoking a model:

```bash
printf '%s\n' '{"hook_event_name":"SessionStart"}' | python3 scripts/agents/hook.py
printf '%s\n' '{"hook_event_name":"PostToolUse","tool_name":"Edit","tool_input":{"file_path":"scripts/agents/hook.py"}}' | python3 scripts/agents/hook.py
```

After activation, edit a small Python file in an agent session and confirm the
hook runs. Run the relevant scientific checks separately using the
[development guide](README.md).

## Maintenance

Change shared skills and role instructions first, then their native loaders
when names or paths change. Keep local settings, transcripts, environment files,
and agent caches out of Git. Shared hooks, definitions, and skills belong in Git.
Use [the task template](task-template.md) for longer work; keep task-specific
working notes in an issue, conversation, or explicitly chosen file.

Configuration formats were checked against the installed Codex CLI 0.169.0 and
Claude Code 2.1.327. Official references:

- [Codex skills](https://developers.openai.com/codex/skills)
- [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
- [Codex hooks](https://learn.chatgpt.com/docs/hooks)
- [Claude skills](https://code.claude.com/docs/en/skills)
- [Claude subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude rules](https://code.claude.com/docs/en/memory)
- [Claude hooks](https://code.claude.com/docs/en/hooks)
