'use strict'

// Realign the installed @qvac/fabric package's `name` field with the canonical
// "@qvac/fabric" wherever it was pulled from the temporary GPR alias
// (@tetherto/fabric-mono, or any future @tetherto/* wrapper).
//
// Why: bare-link and cmake-bare derive a bare addon's identity from
// package.json `.name` (@scope/pkg -> scope__pkg). The consumer addons
// (llm-llamacpp, embed-llamacpp) are compiled against the canonical fabric
// artifact, so their baked DT_NEEDED/SONAME is always `qvac__fabric@0.bare`.
// If fabric is left named `@tetherto/fabric-mono`, bare-link emits the runtime
// as `libtetherto__fabric-mono.<ver>.so` and rewrites the wrong DT_NEEDED key,
// leaving `qvac__fabric@0.bare` unresolved on Android/iOS. Renaming the
// installed package back to `@qvac/fabric` keeps the whole toolchain consistent
// across platforms and across new fabric versions, and is a no-op once fabric
// ships under its real `@qvac/fabric` name on npm.

const fs = require('fs')
const path = require('path')

const CANONICAL = '@qvac/fabric'

function normalizePackage (pkgJsonPath) {
  let raw
  try {
    raw = fs.readFileSync(pkgJsonPath, 'utf8')
  } catch {
    return false
  }

  let pkg
  try {
    pkg = JSON.parse(raw)
  } catch {
    return false
  }

  if (pkg.name === CANONICAL) return false

  pkg.name = CANONICAL
  fs.writeFileSync(pkgJsonPath, JSON.stringify(pkg, null, 2) + '\n')
  return true
}

function walk (nodeModulesDir, changed) {
  let entries
  try {
    entries = fs.readdirSync(nodeModulesDir, { withFileTypes: true })
  } catch {
    return
  }

  for (const entry of entries) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue

    const full = path.join(nodeModulesDir, entry.name)

    if (entry.name === '@qvac') {
      const fabricPkg = path.join(full, 'fabric', 'package.json')
      if (fs.existsSync(fabricPkg) && normalizePackage(fabricPkg)) {
        changed.push(fabricPkg)
      }
    }

    // Recurse into nested node_modules (handles non-hoisted installs).
    if (entry.name.startsWith('@')) {
      let scoped
      try {
        scoped = fs.readdirSync(full, { withFileTypes: true })
      } catch {
        scoped = []
      }
      for (const s of scoped) {
        if (s.isDirectory()) walk(path.join(full, s.name, 'node_modules'), changed)
      }
    } else {
      walk(path.join(full, 'node_modules'), changed)
    }
  }
}

const root = process.argv[2] || '.'
const changed = []
walk(path.join(root, 'node_modules'), changed)

if (changed.length === 0) {
  console.log(`[normalize-fabric] no @qvac/fabric rename needed under ${root}`)
} else {
  for (const c of changed) {
    console.log(`[normalize-fabric] rewrote name -> ${CANONICAL} in ${c}`)
  }
}
