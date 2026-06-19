"use strict";

// src/core.ts
function escapeData(value) {
  return value.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
}
function issueCommand(command, message) {
  process.stdout.write(`::${command}::${escapeData(message)}
`);
}
function getInput(name, options) {
  const value = process.env[`INPUT_${name.replace(/ /g, "_").toUpperCase()}`] || "";
  if (options && options.required && !value) {
    throw new Error(`Input required and not supplied: ${name}`);
  }
  return value.trim();
}
function info(message) {
  process.stdout.write(`${message}
`);
}
function warning(message) {
  issueCommand("warning", message);
}
function error(message) {
  issueCommand("error", message);
}
function setFailed(message) {
  process.exitCode = 1;
  error(message);
}

// src/index.ts
var import_child_process2 = require("child_process");

// src/main-provenance.ts
var import_child_process = require("child_process");
var METADATA_BASENAMES = /* @__PURE__ */ new Set([
  "package.json",
  "CHANGELOG.md",
  "NOTICE",
  "models.md",
  // generated lockfiles (data, not executable source)
  "package-lock.json",
  "npm-shrinkwrap.json",
  "yarn.lock",
  "pnpm-lock.yaml",
  "bun.lock",
  "bun.lockb"
]);
function basename(filePath) {
  const parts = filePath.split("/");
  return parts[parts.length - 1] || filePath;
}
function isReleaseMetadataPath(filePath) {
  if (METADATA_BASENAMES.has(basename(filePath))) return true;
  if (filePath.includes("/changelog/")) return true;
  return false;
}
function git(args) {
  return (0, import_child_process.execFileSync)("git", args, {
    stdio: ["ignore", "pipe", "pipe"],
    maxBuffer: 64 * 1024 * 1024
  }).toString().trim();
}
function tryGit(args) {
  try {
    return { ok: true, out: git(args) };
  } catch {
    return { ok: false, out: "" };
  }
}
function checkMainProvenance(headSha, mainRef = "main") {
  const violations = [];
  const remoteMain = `refs/remotes/origin/${mainRef}`;
  const fetched = tryGit(["fetch", "--no-tags", "--quiet", "origin", `${mainRef}:${remoteMain}`]).ok || tryGit(["fetch", "--no-tags", "--quiet", "origin", mainRef]).ok;
  let mainResolved = remoteMain;
  if (!tryGit(["rev-parse", "--verify", "--quiet", `${remoteMain}^{commit}`]).ok) {
    if (fetched && tryGit(["rev-parse", "--verify", "--quiet", "FETCH_HEAD^{commit}"]).ok) {
      mainResolved = "FETCH_HEAD";
    } else {
      violations.push(
        `Unable to resolve origin/${mainRef}; cannot prove the release branch descends from main.`
      );
      return { violations, inspectedExtraCommits: 0 };
    }
  }
  const base = tryGit(["merge-base", mainResolved, headSha]);
  if (!base.ok || !base.out) {
    violations.push(
      `Release branch shares no history with origin/${mainRef} \u2014 it does not descend from main.`
    );
    return { violations, inspectedExtraCommits: 0 };
  }
  const cherry = tryGit(["cherry", mainResolved, headSha]);
  if (!cherry.ok) {
    violations.push(`Failed to compare release branch against origin/${mainRef} (git cherry error).`);
    return { violations, inspectedExtraCommits: 0 };
  }
  const extraCommits = cherry.out.split("\n").map((line) => line.trim()).filter((line) => line.startsWith("+ ")).map((line) => line.slice(2).trim()).filter(Boolean);
  for (const sha of extraCommits) {
    const files = tryGit(["diff-tree", "--no-commit-id", "--name-only", "-r", sha]);
    const changed = files.out.split("\n").map((f) => f.trim()).filter(Boolean);
    const offending = changed.filter((f) => !isReleaseMetadataPath(f));
    if (offending.length) {
      const short = sha.slice(0, 9);
      const subject = tryGit(["show", "-s", "--format=%s", sha]).out || "(unknown)";
      violations.push(
        `Commit ${short} ("${subject}") is not present on origin/${mainRef} and changes non-release files: ${offending.join(", ")}`
      );
    }
  }
  return { violations, inspectedExtraCommits: extraCommits.length };
}

// src/index.ts
var ZERO_SHA = "0000000000000000000000000000000000000000";
var ENFORCE_MAIN_PROVENANCE_DEFAULT = false;
try {
  const baseRef = getInput("base-ref", { required: true });
  const baseSha = getInput("base-sha", { required: false });
  const headSha = getInput("head-sha", { required: true });
  const pkgSlug = getInput("package-slug", { required: true });
  const pkgJsonPath = getInput("package-json-path", { required: true });
  const changelogPath = getInput("changelog-path", { required: true });
  const mainRef = getInput("main-ref", { required: false }) || "main";
  const enforceInput = getInput("enforce", { required: false }).toLowerCase();
  const enforceMainProvenance = enforceInput === "true" || enforceInput !== "false" && ENFORCE_MAIN_PROVENANCE_DEFAULT;
  const isInitialPush = !baseSha || baseSha === ZERO_SHA;
  const errors = [];
  const match = baseRef.match(/^release-(.+)-(\d+\.\d+\.\d+)$/);
  if (!match) {
    errors.push(
      `Invalid release branch name \u2014 expected: release-${pkgSlug}-x.y.z, actual: ${baseRef}`
    );
  }
  let branchVersion = "";
  if (match) {
    const branchPkg = match[1];
    branchVersion = match[2];
    if (branchPkg !== pkgSlug) {
      warning(
        `Package slug mismatch \u2014 branch targets '${branchPkg}', workflow expects '${pkgSlug}'. This is expected for short-name release branches (e.g. release-diffusion-x.y.z).`
      );
    }
  }
  const headPkg = JSON.parse((0, import_child_process2.execSync)(`git show ${headSha}:${pkgJsonPath}`).toString());
  if (branchVersion && headPkg.version !== branchVersion) {
    errors.push(
      `Version mismatch \u2014 branch version: ${branchVersion}, package.json: ${headPkg.version}`
    );
  }
  if (isInitialPush) {
    info("Initial branch push detected (no base SHA) \u2014 skipping changelog check");
  } else {
    const changedFiles = (0, import_child_process2.execSync)(
      `git diff --name-only ${baseSha} ${headSha}`
    ).toString();
    if (!changedFiles.includes(changelogPath)) {
      errors.push(
        `Missing CHANGELOG update \u2014 file not modified: ${changelogPath}`
      );
    }
  }
  try {
    const { violations, inspectedExtraCommits } = checkMainProvenance(headSha, mainRef);
    if (violations.length) {
      const header = `Release branch \u2194 main provenance: ${violations.length} issue(s) \u2014 the release branch carries content that is not on origin/${mainRef}:`;
      const body = violations.map((v) => `  - ${v}`).join("\n");
      if (enforceMainProvenance) {
        errors.push(`${header}
${body}`);
      } else {
        warning(
          `${header}
${body}
(warn-first: not blocking publish yet \u2014 set enforce=true once validated)`
        );
      }
    } else {
      info(
        `Release branch \u2194 main provenance OK \u2014 ${inspectedExtraCommits} release-only commit(s), no un-merged code relative to origin/${mainRef}`
      );
    }
  } catch (provErr) {
    warning(
      `Release branch \u2194 main provenance check could not complete: ${provErr instanceof Error ? provErr.message : String(provErr)}`
    );
  }
  for (const err of errors) {
    error(err);
  }
  if (errors.length) {
    setFailed(`Release merge guard failed with ${errors.length} error(s):
${errors.join("\n")}`);
  } else {
    info("Release merge guard passed \u2014 branch name, version, and changelog all valid");
  }
} catch (err) {
  setFailed(err instanceof Error ? err.message : String(err));
}
