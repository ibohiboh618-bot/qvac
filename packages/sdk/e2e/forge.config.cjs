"use strict";

const fs = require("node:fs");
const path = require("node:path");
const QvacForgePlugin = require("@qvac/sdk/electron-forge");

const projectDir = __dirname;
const electronConfigPath = path.join(projectDir, "fixtures", "qvac.config.electron.json");
const runtimeConfigPath = path.join(projectDir, "qvac.config.json");
const generatedWorkerDir = path.join(projectDir, "qvac");
const backupDir = path.join(projectDir, ".qvac-worker-backup");
const backupWorkerDir = path.join(backupDir, "qvac");
const backupConfigPath = path.join(backupDir, "qvac.config.json");

// Electron overwrites the shared qvac/worker.entry.mjs + qvac.config.json to package
// its own bundle. Snapshot what's there first and restore it in postPackage, so desktop
// and Electron runs don't clobber each other's worker bundle locally.
function snapshotExistingWorker() {
  fs.rmSync(backupDir, { recursive: true, force: true });
  if (!fs.existsSync(generatedWorkerDir) && !fs.existsSync(runtimeConfigPath)) return;

  fs.mkdirSync(backupDir, { recursive: true });
  if (fs.existsSync(generatedWorkerDir)) {
    fs.cpSync(generatedWorkerDir, backupWorkerDir, { recursive: true });
  }
  if (fs.existsSync(runtimeConfigPath)) {
    fs.copyFileSync(runtimeConfigPath, backupConfigPath);
  }
}

function restorePreviousWorker() {
  fs.rmSync(generatedWorkerDir, { recursive: true, force: true });
  fs.rmSync(runtimeConfigPath, { force: true });

  if (fs.existsSync(backupWorkerDir)) {
    fs.cpSync(backupWorkerDir, generatedWorkerDir, { recursive: true });
  }
  if (fs.existsSync(backupConfigPath)) {
    fs.copyFileSync(backupConfigPath, runtimeConfigPath);
  }
  fs.rmSync(backupDir, { recursive: true, force: true });
}

function ensureRuntimeConfig() {
  fs.copyFileSync(electronConfigPath, runtimeConfigPath);
}

snapshotExistingWorker();
ensureRuntimeConfig();

module.exports = {
  packagerConfig: {
    name: "QVACSDKElectronE2E",
    asar: false,
    ignore: [
      /^\/build\//, // mobile Pods/Gradle output — breaks packager on xcframework code-signature files
      /^\/reports\//, // test run reports
      /^\/\.env$/, // MQTT creds — read by the CLI orchestrator, not the packaged app
      /^\/\.env\.bak-/, // env backups
      /^\/\.qvac-worker-backup\//, // desktop worker snapshot, restored in postPackage
    ],
  },
  rebuildConfig: {},
  makers: [],
  hooks: {
    postPackage: async () => {
      restorePreviousWorker();
    },
  },
  plugins: [
    new QvacForgePlugin({
      projectDir,
      configPath: electronConfigPath,
      logLevel: "debug",
    }),
  ],
};
