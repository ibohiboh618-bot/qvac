import * as core from './core'
import { execSync } from 'child_process'
import { checkMainProvenance } from './main-provenance'

const ZERO_SHA = '0000000000000000000000000000000000000000'

// Release-branch ↔ main provenance check rollout posture. While false the check
// only warns (audit mode) so a brand-new rule never blocks an in-flight release;
// flip to true (and rebuild dist) to make injected, non-main content fail the
// guard — which skips the npm publish via the existing `release-merge-guard`
// gate. Can also be forced per-run via the `enforce` action input.
const ENFORCE_MAIN_PROVENANCE_DEFAULT = false

try {
  const baseRef = core.getInput('base-ref', { required: true })
  const baseSha = core.getInput('base-sha', { required: false })
  const headSha = core.getInput('head-sha', { required: true })
  const pkgSlug = core.getInput('package-slug', { required: true })
  const pkgJsonPath = core.getInput('package-json-path', { required: true })
  const changelogPath = core.getInput('changelog-path', { required: true })
  const mainRef = core.getInput('main-ref', { required: false }) || 'main'
  const enforceInput = core.getInput('enforce', { required: false }).toLowerCase()
  const enforceMainProvenance =
    enforceInput === 'true' || (enforceInput !== 'false' && ENFORCE_MAIN_PROVENANCE_DEFAULT)

  const isInitialPush = !baseSha || baseSha === ZERO_SHA

  const errors: string[] = []

  // ── Branch name validation (always runs)
  const match = baseRef.match(/^release-(.+)-(\d+\.\d+\.\d+)$/)
  if (!match) {
    errors.push(
      `Invalid release branch name — expected: release-${pkgSlug}-x.y.z, actual: ${baseRef}`
    )
  }

  let branchVersion = ''

  if (match) {
    const branchPkg = match[1]
    branchVersion = match[2]

    if (branchPkg !== pkgSlug) {
      core.warning(
        `Package slug mismatch — branch targets '${branchPkg}', workflow expects '${pkgSlug}'. ` +
        `This is expected for short-name release branches (e.g. release-diffusion-x.y.z).`
      )
    }
  }

  // ── package.json version must match the branch version (always runs)
  const headPkg = JSON.parse(execSync(`git show ${headSha}:${pkgJsonPath}`).toString())

  if (branchVersion && headPkg.version !== branchVersion) {
    errors.push(
      `Version mismatch — branch version: ${branchVersion}, package.json: ${headPkg.version}`
    )
  }

  // ── Changelog must be modified in this push (skipped on initial branch creation)
  if (isInitialPush) {
    core.info('Initial branch push detected (no base SHA) — skipping changelog check')
  } else {
    const changedFiles = execSync(
      `git diff --name-only ${baseSha} ${headSha}`
    ).toString()

    if (!changedFiles.includes(changelogPath)) {
      errors.push(
        `Missing CHANGELOG update — file not modified: ${changelogPath}`
      )
    }
  }

  // ── Release branch must descend from main with a metadata-only delta.
  // Runs on every push, including the initial branch-creation push (the very
  // event the changelog check above skips) — that is the local-create-then-push
  // injection vector this guard exists to close. Self-contained: any internal
  // error degrades to a warning so an infra hiccup never blocks a publish.
  try {
    const { violations, inspectedExtraCommits } = checkMainProvenance(headSha, mainRef)
    if (violations.length) {
      const header =
        `Release branch ↔ main provenance: ${violations.length} issue(s) — the release ` +
        `branch carries content that is not on origin/${mainRef}:`
      const body = violations.map((v) => `  - ${v}`).join('\n')
      if (enforceMainProvenance) {
        errors.push(`${header}\n${body}`)
      } else {
        core.warning(
          `${header}\n${body}\n` +
            '(warn-first: not blocking publish yet — set enforce=true once validated)'
        )
      }
    } else {
      core.info(
        `Release branch ↔ main provenance OK — ${inspectedExtraCommits} release-only commit(s), ` +
          `no un-merged code relative to origin/${mainRef}`
      )
    }
  } catch (provErr) {
    core.warning(
      `Release branch ↔ main provenance check could not complete: ` +
        `${provErr instanceof Error ? provErr.message : String(provErr)}`
    )
  }

  // ── Report results
  for (const err of errors) {
    core.error(err)
  }

  if (errors.length) {
    core.setFailed(`Release merge guard failed with ${errors.length} error(s):\n${errors.join('\n')}`)
  } else {
    core.info('Release merge guard passed — branch name, version, and changelog all valid')
  }
} catch (err) {
  core.setFailed(err instanceof Error ? err.message : String(err))
}
