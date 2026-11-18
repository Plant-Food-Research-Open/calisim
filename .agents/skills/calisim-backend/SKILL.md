---
name: calisim-backend
description: Add or modify calisim scientific backends, task specifications, engine registration, or shared workflow dispatch.
---

# Backend development

Read [AGENTS.md](../../../AGENTS.md), the applicable subtree instructions, and
[the architecture map](../../../docs/development/architecture.md). Use an
existing wrapper for the same task as the main integration example.

## Establish the contract

Identify the public task class, specification subclass, lazy engine mapping,
wrapper, functional test module, example, and API reference. Verify backend API
calls against the installed or locked version.

Specify parameter types and ordering, bounds, batching, expected output shapes,
and whether the objective returns simulator output, a discrepancy, or a
log-likelihood. Preserve public defaults and exports unless changing them is
part of the task.

## Implement and verify

Keep library-specific conversions inside the wrapper and use shared workflow
helpers for objective calls, hooks, artifacts, and estimates. Register built-in
engines through a `module:Class` mapping; preserve external entry-point loading.
Keep optional imports behind the relevant engine boundary.

Add focused coverage for observable behavior and the relevant backend workflow.
For a new engine, check unsupported methods and parameter or output cases that
matter to its contract. Exercise changed shared behavior through affected
callers. Choose small cases and justified tolerances.

When dependencies change, use Poetry to update declarations and the lockfile;
identify optional extras and the corresponding third-party license. Update the
example and public documentation where behavior changes.

Use [the development guide](../../../docs/development/README.md) for commands.
Report the files changed, exact checks and results, compatibility decisions, and
any scientific or environment assumptions still needing review.
