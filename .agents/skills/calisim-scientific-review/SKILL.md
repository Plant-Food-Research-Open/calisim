---
name: calisim-scientific-review
description: Review calisim changes for numerical correctness, calibration contracts, objective meaning, reproducibility, and scientific evidence.
---

# Scientific review

Keep this workflow read-only unless the user also asks for implementation.
Read [AGENTS.md](../../../AGENTS.md), applicable subtree rules, and the supplied
diff or files. Trace changed behavior into actual callers and tests.

Concentrate on checks relevant to the change:

- Parameter ordering, types, constants, categorical conversion, bounds, units,
  prior or distribution conventions, and simulator input contracts.
- Array shapes, batching, observed/simulated alignment, output labels, and
  scalar versus multi-objective results.
- Discrepancy minimization versus likelihood or log-likelihood maximization,
  sign conventions, and any backend-specific transforms.
- Seeds for each stochastic backend, generator state, justified tolerances,
  convergence assumptions, and numerical stability.
- Data/model provenance and whether an example, rendered report, or test run
  actually supports a stated scientific claim.
- Lazy imports, optional dependencies, workflow hooks, and public compatibility
  where they affect scientific execution.

Use installed or locked library APIs as evidence. If an experiment is needed,
describe the smallest useful case and its expected invariant; run it only when
the task and available tools permit it. Do not claim convergence or scientific
validity from a syntax check or a passing smoke test.

Return actionable findings ordered by impact. Include a file/line reference,
the concrete failure scenario, and the evidence or test that exposes it.
Separate verified findings from questions and limitations. If no findings are
identified, say so and state which behavior was not verified.
