/**
 * Patched ios/link.mjs for manifest-aware native addon linking.
 *
 * If qvac/addons.manifest.json exists, only links the allowlisted addons.
 * Otherwise, falls back to linking all installed addons.
 *
 * This file is copied over react-native-bare-kit/ios/link.mjs
 * by withMobileBundle.ts during expo prebuild.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import link from "bare-link";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, "..", "..", "..");
const addonsDir = path.join(__dirname, "addons");

// bare-link derives a bare addon's identity from package.json `.name`. When
// @qvac/fabric is pulled from GPR (as @tetherto/fabric-mono, or any future
// @tetherto/* wrapper) the rename makes bare-link emit the runtime as
// `libtetherto__fabric-mono.*.so` and rewrite the wrong DT_NEEDED, leaving the
// canonical `qvac__fabric@0.bare` that consumer addons bake as their dependency
// unresolved on-device. Realign the installed fabric `.name` with the canonical
// `@qvac/fabric` before linking. No-op once fabric ships under @qvac on npm.
normalizeFabricName(projectRoot);

if (fs.existsSync(addonsDir)) {
  console.log("[QVAC] Cleaning existing addons directory...");
  fs.rmSync(addonsDir, { recursive: true, force: true });
}

const manifestPath = path.join(projectRoot, "qvac", "addons.manifest.json");

let pkg = null;
if (fs.existsSync(manifestPath)) {
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const addons = Array.isArray(manifest.addons) ? manifest.addons : [];

    if (addons.length > 0) {
      console.log(
        `[QVAC] Using addons manifest (${addons.length} addons): ${addons.join(", ")}`,
      );
      pkg = {
        name: "qvac-addon-linker",
        version: "0.0.0",
        dependencies: Object.fromEntries(addons.map((name) => [name, "*"])),
      };
    } else {
      console.log("[QVAC] Addons manifest is empty, linking all addons");
    }
  } catch (err) {
    console.warn(
      "[QVAC] Failed to parse addons manifest, linking all addons:",
      err.message,
    );
  }
} else {
  console.log("[QVAC] No addons manifest found, linking all addons");
}

for await (const resource of link(
  projectRoot,
  {
    hosts: ["ios-arm64", "ios-arm64-simulator", "ios-x64-simulator"],
    out: addonsDir,
  },
  pkg,
)) {
  console.log("Wrote", resource);
}

function normalizeFabricName(root) {
  const CANONICAL = "@qvac/fabric";

  function rewrite(pkgJsonPath) {
    let pkg;
    try {
      pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf8"));
    } catch {
      return;
    }
    if (pkg.name === CANONICAL) return;
    pkg.name = CANONICAL;
    fs.writeFileSync(pkgJsonPath, JSON.stringify(pkg, null, 2) + "\n");
    console.log(`[QVAC] Normalized @qvac/fabric name in ${pkgJsonPath}`);
  }

  function walk(nodeModulesDir) {
    let entries;
    try {
      entries = fs.readdirSync(nodeModulesDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
      const full = path.join(nodeModulesDir, entry.name);
      if (entry.name === "@qvac") {
        const fabricPkg = path.join(full, "fabric", "package.json");
        if (fs.existsSync(fabricPkg)) rewrite(fabricPkg);
      }
      if (entry.name.startsWith("@")) {
        let scoped;
        try {
          scoped = fs.readdirSync(full, { withFileTypes: true });
        } catch {
          scoped = [];
        }
        for (const s of scoped) {
          if (s.isDirectory()) walk(path.join(full, s.name, "node_modules"));
        }
      } else {
        walk(path.join(full, "node_modules"));
      }
    }
  }

  walk(path.join(root, "node_modules"));
}
