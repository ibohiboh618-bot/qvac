import { execFileSync } from 'child_process'

// Files a release branch is allowed to add/change relative to main without that
// content existing on main. These are the release-only artifacts the gitflow
// permits a Release PR to introduce (version bump, changelog, attributions,
// generated lockfiles, model lists). Everything else — i.e. real source/code —
// MUST already be on main (or be a patch-id-equivalent cherry-pick of a commit
// that is), otherwise it is content that never went through main review.
const METADATA_BASENAMES = new Set([
  'package.json',
  'CHANGELOG.md',
  'NOTICE',
  'models.md',
  // generated lockfiles (data, not executable source)
  'package-lock.json',
  'npm-shrinkwrap.json',
  'yarn.lock',
  'pnpm-lock.yaml',
  'bun.lock',
  'bun.lockb'
])

function basename(filePath: string): string {
  const parts = filePath.split('/')
  return parts[parts.length - 1] || filePath
}

export function isReleaseMetadataPath(filePath: string): boolean {
  if (METADATA_BASENAMES.has(basename(filePath))) return true
  // Per-version changelog folders: packages/<pkg>/changelog/<x.y.z>/...
  if (filePath.includes('/changelog/')) return true
  return false
}

function git(args: string[]): string {
  return execFileSync('git', args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024
  })
    .toString()
    .trim()
}

function tryGit(args: string[]): { ok: boolean; out: string } {
  try {
    return { ok: true, out: git(args) }
  } catch {
    return { ok: false, out: '' }
  }
}

export interface ProvenanceResult {
  violations: string[]
  inspectedExtraCommits: number
}

/**
 * Verify that a release branch descends from `main` and that every commit it
 * carries which is NOT already on `main` (by patch-id, so legitimate
 * cherry-picked patches are recognised as already-merged) touches release
 * metadata only.
 *
 * This closes the gap where a `release-*` branch can be created locally with
 * injected, never-reviewed content and pushed straight to the publish path:
 * the existing guard validates branch name / version / changelog but never
 * proves the branch's code actually came from `main`.
 *
 * Pure detection: returns the list of violations. The caller decides whether to
 * warn (audit) or fail (enforce).
 */
export function checkMainProvenance(headSha: string, mainRef = 'main'): ProvenanceResult {
  const violations: string[] = []
  const remoteMain = `refs/remotes/origin/${mainRef}`

  // Make origin/<main> available locally. Workflows check out the release branch
  // with fetch-depth: 0 but do not necessarily fetch main as a tracking ref.
  const fetched =
    tryGit(['fetch', '--no-tags', '--quiet', 'origin', `${mainRef}:${remoteMain}`]).ok ||
    tryGit(['fetch', '--no-tags', '--quiet', 'origin', mainRef]).ok

  // Resolve a usable ref for main: prefer the tracking ref, fall back to
  // FETCH_HEAD (set by the bare `git fetch origin <main>` above).
  let mainResolved = remoteMain
  if (!tryGit(['rev-parse', '--verify', '--quiet', `${remoteMain}^{commit}`]).ok) {
    if (fetched && tryGit(['rev-parse', '--verify', '--quiet', 'FETCH_HEAD^{commit}']).ok) {
      mainResolved = 'FETCH_HEAD'
    } else {
      violations.push(
        `Unable to resolve origin/${mainRef}; cannot prove the release branch descends from main.`
      )
      return { violations, inspectedExtraCommits: 0 }
    }
  }

  // The branch must share history with main at all.
  const base = tryGit(['merge-base', mainResolved, headSha])
  if (!base.ok || !base.out) {
    violations.push(
      `Release branch shares no history with origin/${mainRef} — it does not descend from main.`
    )
    return { violations, inspectedExtraCommits: 0 }
  }

  // Commits reachable from HEAD that are not already on main, by patch-id.
  // `git cherry` marks each with '+' (no equivalent on main) or '-' (equivalent
  // patch already on main, e.g. a clean cherry-pick). Merge commits are omitted.
  const cherry = tryGit(['cherry', mainResolved, headSha])
  if (!cherry.ok) {
    violations.push(`Failed to compare release branch against origin/${mainRef} (git cherry error).`)
    return { violations, inspectedExtraCommits: 0 }
  }

  const extraCommits = cherry.out
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('+ '))
    .map((line) => line.slice(2).trim())
    .filter(Boolean)

  for (const sha of extraCommits) {
    const files = tryGit(['diff-tree', '--no-commit-id', '--name-only', '-r', sha])
    const changed = files.out
      .split('\n')
      .map((f) => f.trim())
      .filter(Boolean)
    const offending = changed.filter((f) => !isReleaseMetadataPath(f))
    if (offending.length) {
      const short = sha.slice(0, 9)
      const subject = tryGit(['show', '-s', '--format=%s', sha]).out || '(unknown)'
      violations.push(
        `Commit ${short} ("${subject}") is not present on origin/${mainRef} and changes ` +
          `non-release files: ${offending.join(', ')}`
      )
    }
  }

  return { violations, inspectedExtraCommits: extraCommits.length }
}
