# QVAC SDK v0.12.3 Release Notes

📦 **NPM:** https://www.npmjs.com/package/@qvac/sdk/v/0.12.3

This is a dependency-maintenance patch. `@qvac/sdk` and `@qvac/bare-sdk` adopt `bare-fetch` 3.x and the SDK moves its dev-only `bare-subprocess` to 6.x. There are no source or public API changes.

## Maintenance

Bump bare-fetch to ^3.0.1 (adopt 3.x; public fetch API unchanged). Bump dev bare-subprocess to ^6.1.0.

The bare-fetch 2→3 jump is a transitive-only major: the public fetch API is unchanged. The only 3.x behavior change is 3.0.1 header validation, and every SDK header construction (`new Headers({ 'User-Agent': ... })`, `append('Range', 'bytes=…')`) already builds RFC-valid headers. The bare-subprocess 5→6 bump is dev-only and affects no shipped code.
