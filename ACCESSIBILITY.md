# Accessibility

We aim to make this project's software, documentation, examples, and other
materials accessible to people with different abilities, devices, and ways of
working. Accessibility is part of how we design, maintain, and review the project.

This policy describes our goals, contributor expectations, and approach to
reporting and addressing barriers. Apply the guidance relevant to the project's
features and outputs.

## Accessibility goals

For web content and interfaces, we work toward
[WCAG 2.2 Level AA](https://www.w3.org/TR/WCAG22/). This is a development goal;
any claim of conformance must be supported by an evaluation that identifies its
scope, date, methods, findings, and remaining limitations.

Across the project, we prioritise:

- Clear language, descriptive links, and a logical content structure.
- Keyboard access and visible focus for interactive interfaces.
- Meaningful labels, headings, and semantics for assistive technologies.
- Readable content that works with zoom, different screen sizes, and user settings.
- Equivalent ways to understand information presented through images, colour,
  audio, video, or interaction.
- Clear instructions, actionable errors, and accessible examples and outputs.

## Contributor expectations

Consider accessibility when adding or changing user-facing content or behaviour.
Apply these practices where relevant:

- Use a logical heading hierarchy, semantic lists, and descriptive link text.
- Provide useful alternative text for informative images. For complex figures,
  include a nearby explanation or an accessible table of the relevant data.
  See the [W3C alternative-text decision tree](https://www.w3.org/WAI/tutorials/images/decision-tree/).
- Give tables appropriate headers and explain units, abbreviations, and symbols.
- Use sufficient contrast and additional cues alongside colour to convey meaning.
- Provide captions for video and transcripts for audio, with descriptions of
  essential visual information where needed.
- Keep text and controls usable with zoom and on narrow screens.
- Give controls accessible names and states. Keep focus order logical, make
  actions work with a keyboard, and avoid keyboard traps.
- Declare the language of web content and identify changes of language when needed.
- Make instructions, code examples, command-line help, and error messages clear
  and usable without relying solely on screenshots or visual layout.

### Reviewing changes

Use existing project checks where available. For affected interfaces, combine
automated checks with manual keyboard testing and, where possible, checks with
assistive technology. Automated tools cover only part of an accessibility review.

In a pull request, describe what was checked, the tools and environments used,
any barriers found, and any checks that could not be completed. Record unresolved
issues and workarounds so maintainers can assess their impact. Introduce or change
automated checks through the project's normal contribution process.

## Reporting an accessibility barrier

Open an issue in the repository's issue tracker. Use an accessibility template or
label if one is available; a general issue is also welcome. If the tracker is
unavailable or difficult to use, use a contact or support route documented in the
repository's README.

Include whatever information you can comfortably provide:

- The task you were trying to complete and the barrier you encountered.
- The affected page, file, command, feature, or output, including a version if known.
- Steps to reproduce the problem and what you expected to happen.
- Relevant browser, operating system, device, or assistive technology details.
- The impact on your work and any workaround you have found.

Screenshots and recordings are optional. You do not need to disclose a disability,
diagnosis, or other personal information to report a barrier or request help.
Avoid including confidential data or credentials in a public report.

## Prioritisation and response

Repository maintainers coordinate accessibility work and may delegate individual
issues to contributors. Reports should be acknowledged respectfully and assessed
according to their effect on users' ability to complete tasks.

Prioritise barriers that prevent essential tasks or have no practical workaround.
Consider the number of affected users, available alternatives, dependencies, and
the effort needed to address an issue. Reporters do not need to assign a severity.

Track agreed next steps and, where possible, provide workarounds and progress
updates. Set response or resolution targets only when the project can support
them, and communicate changes to an agreed timeline. Invite feedback on whether
a proposed fix resolves the reported barrier.

## Tested environments and known limitations

Record the browsers, devices, assistive technologies, software versions, and
output formats actually checked in project documentation, issues, or pull
requests. Describe the scope and date of testing when reporting compatibility.

Document confirmed barriers with their user impact, affected versions, known
workarounds, and follow-up work. Identify relevant limits in third-party tools
or embedded content. Revisit these records as the project and its dependencies
change.

## Maintaining this policy

Maintainers should review this policy when user-facing features, support routes,
or accessibility practices change. Suggestions and corrections are welcome
through the repository's usual contribution and support channels.
