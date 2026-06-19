/*
 * Integration test for the release-merge-guard action's main-provenance check.
 *
 * Runs the *compiled* action (dist/index.js) against throwaway git repos that
 * reproduce real release scenarios, asserting the warn-first / enforce
 * behaviour. Uses only Node built-ins + the committed bundle, so it runs on a
 * bare runner with no `npm install` (the bundle is self-contained).
 *
 * Run locally:   node .github/actions/release-merge-guard/test/provenance.test.cjs
 * Run via npm:   npm test   (from this action directory)
 */
const { execFileSync } = require('child_process')
const fs = require('fs')
const os = require('os')
const path = require('path')

const DIST = path.resolve(__dirname, '..', 'dist', 'index.js')

function git(cwd, args) {
  return execFileSync('git', args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] }).toString().trim()
}

function commit(cwd, message) {
  git(cwd, ['add', '-A'])
  git(cwd, ['-c', 'user.email=guard@test', '-c', 'user.name=guard', 'commit', '--no-gpg-sign', '-m', message])
  return git(cwd, ['rev-parse', 'HEAD'])
}

function cherryPick(cwd, sha) {
  git(cwd, ['-c', 'user.email=guard@test', '-c', 'user.name=guard', 'cherry-pick', '--no-gpg-sign', sha])
}

function write(cwd, rel, content) {
  const p = path.join(cwd, rel)
  fs.mkdirSync(path.dirname(p), { recursive: true })
  fs.writeFileSync(p, content)
}

function pkg(version) {
  return JSON.stringify({ name: 'demo', version }, null, 2) + '\n'
}

// Fresh repo with a `main` pushed to a local bare "origin".
function setupRepo() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'guard-'))
  const bare = path.join(root, 'remote.git')
  const work = path.join(root, 'work')
  fs.mkdirSync(bare)
  git(bare, ['init', '--bare', '-q'])
  fs.mkdirSync(work)
  git(work, ['init', '-q'])
  git(work, ['symbolic-ref', 'HEAD', 'refs/heads/main'])
  git(work, ['remote', 'add', 'origin', bare])
  write(work, 'packages/demo/package.json', pkg('1.0.0'))
  write(work, 'packages/demo/src/app.js', 'module.exports = 1\n')
  write(work, 'packages/demo/CHANGELOG.md', '# Changelog\n')
  commit(work, 'base')
  git(work, ['push', '-q', '-u', 'origin', 'main'])
  return work
}

function runAction(work, { baseRef, headSha, enforce = '' }) {
  const env = {
    ...process.env,
    'INPUT_BASE-REF': baseRef,
    // Empty base-sha => "initial push" => changelog check skipped, isolating the
    // provenance check as the only variable under test.
    'INPUT_BASE-SHA': '0000000000000000000000000000000000000000',
    'INPUT_HEAD-SHA': headSha,
    'INPUT_PACKAGE-SLUG': 'demo',
    'INPUT_PACKAGE-JSON-PATH': 'packages/demo/package.json',
    'INPUT_CHANGELOG-PATH': 'packages/demo/CHANGELOG.md',
    'INPUT_MAIN-REF': 'main',
    'INPUT_ENFORCE': enforce
  }
  try {
    const out = execFileSync('node', [DIST], {
      cwd: work,
      env,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    })
    return { status: 0, out }
  } catch (e) {
    return { status: e.status == null ? 1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` }
  }
}

let failures = 0
function assert(name, cond, detail) {
  const ok = !!cond
  if (!ok) failures++
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n        -> ${detail}`}`)
}

// 1. Clean release: cut from main + a metadata-only (version) commit -> passes.
;(() => {
  const work = setupRepo()
  git(work, ['checkout', '-q', '-b', 'release-demo-1.2.3', 'main'])
  write(work, 'packages/demo/package.json', pkg('1.2.3'))
  write(work, 'packages/demo/CHANGELOG.md', '# Changelog\n\n## [1.2.3]\n- note\n')
  const head = commit(work, 'release: 1.2.3')
  const r = runAction(work, { baseRef: 'release-demo-1.2.3', headSha: head })
  assert('clean release: exits 0', r.status === 0, `status=${r.status}\n${r.out}`)
  assert('clean release: provenance OK', /provenance OK/.test(r.out), r.out)
})()

// 2. Injected code, warn-first (default) -> non-blocking warning naming the file.
;(() => {
  const work = setupRepo()
  git(work, ['checkout', '-q', '-b', 'release-demo-1.2.3', 'main'])
  write(work, 'packages/demo/src/app.js', 'module.exports = "backdoor"\n')
  commit(work, 'inject backdoor')
  write(work, 'packages/demo/package.json', pkg('1.2.3'))
  const head = commit(work, 'release: 1.2.3')
  const r = runAction(work, { baseRef: 'release-demo-1.2.3', headSha: head })
  assert('injected/warn: exits 0 (non-blocking)', r.status === 0, `status=${r.status}\n${r.out}`)
  assert(
    'injected/warn: ::warning:: names app.js',
    /::warning::/.test(r.out) && /provenance/.test(r.out) && /app\.js/.test(r.out),
    r.out
  )
})()

// 3. Injected code, enforce=true -> blocks (fails the guard job).
;(() => {
  const work = setupRepo()
  git(work, ['checkout', '-q', '-b', 'release-demo-1.2.3', 'main'])
  write(work, 'packages/demo/src/app.js', 'module.exports = "backdoor"\n')
  commit(work, 'inject backdoor')
  write(work, 'packages/demo/package.json', pkg('1.2.3'))
  const head = commit(work, 'release: 1.2.3')
  const r = runAction(work, { baseRef: 'release-demo-1.2.3', headSha: head, enforce: 'true' })
  assert('injected/enforce: exits non-zero (blocks publish)', r.status === 1, `status=${r.status}\n${r.out}`)
  assert('injected/enforce: ::error:: emitted', /::error::/.test(r.out), r.out)
})()

// 4. Patch cherry-pick (fix already on main, even if cut from an older commit) -> passes.
;(() => {
  const work = setupRepo()
  const base = git(work, ['rev-parse', 'HEAD'])
  write(work, 'packages/demo/src/app.js', 'module.exports = 2 // fix\n')
  const fix = commit(work, 'fix: value')
  git(work, ['push', '-q', 'origin', 'main'])
  git(work, ['checkout', '-q', '-b', 'release-demo-1.0.1', base])
  cherryPick(work, fix)
  write(work, 'packages/demo/package.json', pkg('1.0.1'))
  const head = commit(work, 'release: 1.0.1')
  const r = runAction(work, { baseRef: 'release-demo-1.0.1', headSha: head })
  assert('patch cherry-pick: exits 0', r.status === 0, `status=${r.status}\n${r.out}`)
  assert('patch cherry-pick: provenance OK', /provenance OK/.test(r.out), r.out)
})()

// 5. Orphan branch sharing no history with main -> flagged (warn-first).
;(() => {
  const work = setupRepo()
  git(work, ['checkout', '-q', '--orphan', 'release-demo-9.9.9'])
  git(work, ['rm', '-rfq', '--ignore-unmatch', '.'])
  write(work, 'packages/demo/package.json', pkg('9.9.9'))
  write(work, 'packages/demo/CHANGELOG.md', '# Changelog\n\n## [9.9.9]\n- x\n')
  write(work, 'packages/demo/src/evil.js', 'evil\n')
  const head = commit(work, 'orphan release')
  const r = runAction(work, { baseRef: 'release-demo-9.9.9', headSha: head })
  assert('orphan: exits 0 (warn-first)', r.status === 0, `status=${r.status}\n${r.out}`)
  assert('orphan: ::warning:: "no history"', /::warning::/.test(r.out) && /no history/i.test(r.out), r.out)
})()

console.log(`\n${failures === 0 ? 'ALL PASS' : failures + ' FAILURE(S)'}`)
process.exit(failures === 0 ? 0 : 1)
