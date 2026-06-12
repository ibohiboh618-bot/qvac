'use strict'
// QVAC-19371 (A1 contract): SOURCES — the builds-under-comparison axis.
// A "source" is one thing being measured: our addon at some version, or one of
// the native CLIs (fabric = the company llama.cpp fork, upstream = original
// llama.cpp). Selected per run via the `matrix_sources` workflow input
// (tokens: addon | addon@candidate | addon@baseline | fabric@<ref> | upstream@<ref>).
//
// OWNERSHIP: runner workstream (Dev A). The report side never imports this —
// it sees sources only as the `source_id`/`source_ref` marker fields.
//
// A1 ships the contract + the resolver for what runs TODAY (the published
// addon). The remaining types are wired by their subtasks:
//   addon@candidate  — prebuild built from the PR ref            TODO(A2)
//   addon@baseline   — pinned npm version (per-model/baseline)   TODO(A2)
//   fabric/upstream  — already runnable via the several-sources
//                      CLI path; arbitrary-commit refs + SHA-keyed
//                      build cache                                TODO(A5)

// Parse one matrix_sources token into { id, type, ref }.
function parseSourceToken (token) {
  const t = String(token || '').trim()
  if (!t) return null
  const at = t.indexOf('@')
  const kind = at === -1 ? t : t.slice(0, at)
  const ref = at === -1 ? '' : t.slice(at + 1)
  switch (kind) {
    case 'addon':
      if (ref && ref !== 'candidate' && ref !== 'baseline') {
        throw new Error(`addon source ref must be 'candidate' or 'baseline' (got '${t}')`)
      }
      return { id: t, type: 'addon', ref: ref || 'published' }
    case 'fabric':
      return { id: t, type: 'fabric-cli', ref: ref || 'v8189.0.2' }
    case 'upstream':
      return { id: t, type: 'upstream-cli', ref: ref || 'b8189' }
    default:
      throw new Error(`unknown source '${t}' (known: addon[@candidate|@baseline], fabric@<ref>, upstream@<ref>)`)
  }
}

function parseSources (raw) {
  return String(raw || '').split(',').map(parseSourceToken).filter(Boolean)
}

// Resolve an addon source to the prebuilds directory the harness should load.
// TODO(A2): candidate → the PR-ref prebuild artifact dir; baseline → the
// pinned-npm prebuild dir. Today both fall through to the workspace default.
function addonPrebuildDir (source, workdir) {
  return `${workdir}/prebuilds`
}

module.exports = { parseSourceToken, parseSources, addonPrebuildDir }
