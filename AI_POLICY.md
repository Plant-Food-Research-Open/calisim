# AI-assisted development policy

This project permits AI-assisted and agentic work on code, documentation, tests,
reviews, and development tooling. Contributions must meet the project's quality,
security, licensing, and contribution requirements regardless of how they were
produced.

## Responsibility and review

Contributors remain responsible for the accuracy and suitability of the work
they submit. AI-generated contributions follow the same review and acceptance
process as other contributions. A human contributor must take responsibility
for the submission and complete any human review required by the project.

State whether human review is complete or pending. Agent-run checks and automated
reviews provide evidence for reviewers; they do not constitute human review.

## Agent scope and authorisation

Agents may implement assigned tasks, run appropriate checks, prepare changes,
and carry out other actions covered by the contributor's instructions. Follow
repository-specific guidance, including agent instructions, contribution rules,
and security policies where these are provided.

Respect the task's scope and preserve existing work. A review-only request does
not authorise edits. Keep unrelated changes separate, and identify assumptions
or missing information that materially affect the work.

Commits, external messages, merges, deployments, releases, and other shared
changes must follow the contributor's authorisation and the project's approval
process. Existing authorisation applies to actions within its scope. Do not infer
permission to perform unrelated or destructive actions from access to a tool.

## Disclosing material AI assistance

Disclose material AI assistance where the work is submitted, such as a pull
request description, issue, review, or other contribution record. Briefly state:

- Which tool or service was used and what it helped produce or review.
- Which checks were performed and their results.
- Remaining limitations and whether human review is complete or pending.

For example: "An AI coding assistant helped implement the change. Unit tests
pass; integration testing and human review are pending."

Keep disclosure relevant to the contribution. Full chat transcripts, prompts,
and large blocks of raw output are not required. Review any material shared
publicly for confidential information or credentials.

## Verification and evidence

Check generated code, explanations, references, and API usage against the
project's actual environment and appropriate sources. Do not invent results,
citations, provenance, test outcomes, or security and privacy guarantees.
Distinguish proposed work from implemented and verified behaviour.

Run checks appropriate to the change. Report failures, skipped checks, missing
dependencies, and untested environments accurately. Syntax checks, tests, builds,
rendered documents, and executed workflows provide different evidence; identify
which checks actually ran and what they establish.

For work involving data, models, or analytical results, preserve source inputs
and record the versions, parameters, transformations, and environment needed to
understand or reproduce the result. Explain substantive assumptions and use
appropriate validation for the project's domain.

Keep dependency changes deliberate and review generated files and configuration
for effects beyond the assigned task. Follow the project's established testing,
formatting, accessibility, and review practices.

## Data protection and confidentiality

Use only data and services approved for the task. Share the minimum context
needed, and check applicable organisational rules, data-owner permissions, and
service settings before providing non-public material to an external AI service.

Do not expose credentials, personal information, confidential code, or restricted
data through prompts, logs, generated examples, or contribution records. Use
public, synthetic, or redacted examples where appropriate, and verify that the
chosen example is suitable for sharing.

## Licensing and provenance

Respect software licences, copyright, and attribution requirements when using
AI-generated material or adapting suggested code and text. Verify source and
licensing information when it affects whether an output can be included.
Preserve required notices and document relevant sources or adaptations.

## Contributions and concerns

Follow the repository's contribution and conduct policies where available.
Report suspected vulnerabilities or disclosures of sensitive information through
the project's documented security or private reporting route. Use the normal
support or issue channels for other questions about AI-assisted contributions.

Maintainers should update this policy as the project's tools, risks, and review
practices change. Project-specific requirements may supplement this policy and
should be documented alongside it.
