# Documentation rules

Use the root instructions together with these rules for `docs/`.

- Published Sphinx sources are in `docs/source/`; contributor guidance lives
  in `docs/development/` and is linked from the repository README.
- API references are grouped by task under `docs/source/api_reference/`.
  Update public examples and tutorials when API behavior changes.
- Community contribution and conduct documents have copies under
  `docs/source/community/`. Keep corresponding content consistent.
- Verify commands and scientific claims against source and locked versions.
  Distinguish a rendered page from an executed example or verified analysis.
- Use the [documentation skill](../.agents/skills/calisim-docs/SKILL.md) for
  choosing documentation checks. Keep build output out of patches.
