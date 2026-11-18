# AI-assisted development policy

This workshop welcomes AI-assisted and agentic development, including Codex,
for R code, teaching materials, documentation, tests, and repository tooling.
Contributors remain responsible for the accuracy, reproducibility, and quality
of the work they submit.

## Human responsibility and agent scope

Agents may implement assigned tasks, run relevant checks, and prepare patches or
draft contributions within the contributor's instructions. Follow `AGENTS.md`
and the repository skills when using Codex.

A human contributor must review and take responsibility for a contribution
before it is merged or published. Agent-produced drafts should identify any
human review that is still pending. Do not describe a change as human-reviewed
unless that review has actually happened.

Respect the task's scope and existing work. A review-only request does not
authorise edits. Publishing images, releases, or archived research outputs must
be covered by the contributor's instructions and repository permissions.

## Disclosing AI assistance

Disclose material AI assistance in the pull request description, issue, review,
or comment where the work is submitted. A short statement should identify:

- The tool used and the parts of the work it helped produce.
- The checks actually performed and their results.
- Remaining limitations and whether human review is complete or pending.

For example: "Codex helped revise the workflow configuration. Compose validation
passed; the image has not been built. Human review is pending."

Keep discussion focused on the question or change being reviewed. Read and
verify generated text before presenting it as your own conclusion. Full chat
transcripts, prompts, and large blocks of raw output are not required.

## Scientific and technical verification

Check generated code, explanations, citations, and package APIs against the
project's actual environment and appropriate sources. Do not fabricate research
results, references, test outcomes, dataset provenance, or privacy guarantees.
Distinguish planned teaching content from implemented, tested examples.

Preserve source data and document substantive analytical decisions. Record
dataset and model versions and the software environment when they affect a
result. A syntax check, successful render, image build, and executed analysis
provide different evidence; report which checks ran.

## Research data and privacy

Use public demonstration data for the workshop's privacy exercises. Do not send
restricted research data, credentials, or confidential material to an external
AI service without the relevant data owner's authorisation.

Synthetic observations or added noise do not by themselves establish
differential privacy. Any privacy demonstration must explain its mechanism,
assumptions, and accounting, and distinguish reproducing a workflow from
verifying findings based on inaccessible original data.

## Contributions and concerns

Follow the [contribution guidelines](CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md). Use the [security policy](SECURITY.md)
when reporting a vulnerability or a possible disclosure of sensitive information.
