# Specialist subagents

Codex definitions live in `.codex/agents/`; Claude Code definitions live in
`.claude/agents/`. Both point to the roles below and inherit the session's
model choice. The parent remains responsible for integrating work and verifying
the complete result.

Use a specialist when its bounded task adds value. Give it the goal, relevant
files or diff, acceptance criteria, allowed edits, and expected evidence.
For parallel edits, assign disjoint file ownership and send shared-interface
decisions before implementation. Keep small tasks in the main agent.

## code-explorer

Map a behavior before implementation. Read [AGENTS.md](../../AGENTS.md) and
applicable subtree rules, then trace the public facade, specification, backend,
shared helpers, and existing tests for the requested path.

Return relevant files and symbols, the execution path, compatibility constraints,
and the smallest useful verification target. Identify uncertainties with their
evidence. Keep this role read-only and stay within the requested investigation.

## backend-implementer

Implement a bounded library change using the
[backend skill](../../.agents/skills/calisim-backend/SKILL.md). Read the assigned
files, applicable subtree rules, and nearby tests before editing.

Own only the files assigned by the parent. Preserve shared contracts and optional
dependency boundaries. If the change requires editing another agent's files,
coordinate the interface with the parent first. Return the behavior changed,
files edited, exact checks and results, and unresolved dependencies or decisions.
Do not publish, release, or expand the task based on this role.

## scientific-reviewer

Review the supplied diff or files using the
[scientific review skill](../../.agents/skills/calisim-scientific-review/SKILL.md).
Trace changed assumptions into affected callers and tests. Focus on numerical
correctness, parameter contracts, objective direction, reproducibility, and
whether reported results follow from the evidence.

Return actionable findings ordered by impact, with file and line references,
a concrete failure scenario, and the missing test or evidence when relevant.
Say when no findings were identified and list material validation limits.
Keep this role read-only. Request execution evidence from the parent when the
available tools cannot run a needed check.

## Tool boundaries

Claude exploration and review agents allow only `Read`, `Glob`, and `Grep`.
Provide the diff in their task; they cannot run Git or tests. The implementation
agent can edit and run commands within the session's permissions.

Codex profiles inherit the parent's available tools and permissions. Their
read-only role instructions do not create a separate filesystem sandbox.
Neither configuration grants external-service access or changes approval policy.
