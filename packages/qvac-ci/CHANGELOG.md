# Changelog

## [0.2.0] - 2026-06-24

### Changed

- Removed the `paparam` dependency; CLI parsing now uses an internal zero-dependency module (`lib/cli.js`)
- Help output: usage line omits `[command]` for leaf commands and no longer duplicates flag short names

## [0.1.0] - 2026-06-04

### Added

- Initial release
- `pending-approvals` subcommand: checks PR approval status and posts a review-status comment
- Modular command architecture with `Command` base class and per-command directory layout
- Security model: secrets via env vars only, `redact()` + `sanitizeError()` on all output
