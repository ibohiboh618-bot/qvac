# Secrets inventory (v1)

Version: v1 draft &nbsp;|&nbsp; Last reviewed: 2026-06-02 &nbsp;|&nbsp; Owner: Yauheni
(`yauhenipankratovich-web`) &nbsp;|&nbsp; Security review: Giacomo (`GiacomoSorbiWork`)

A v1 inventory of every secret / token / credential class the team owns. This is the
"what do we have" half of the Q2 deliverable; the cadences and triggers live in
[`secrets-rotation-policy.md`](secrets-rotation-policy.md). See the
[folder README](README.md) for scope and the source-of-truth disclaimer.

> **This file contains identifiers and metadata only — never secret values.** Live
> values and access lists live in GitHub, 1Password, the cloud consoles, Apple Developer
> and Google Play. Fields that cannot be derived from this repository are `unknown` or
> `TBD` and must be completed by the owner of record.

## How to read this inventory

**Captured fields (per row):**

- **Identifier** — the secret's name / key as it appears in its store.
- **Where it lives** — vault path or GitHub UI path.
- **Read/use access** — who can read or use it today.
- **Owner of record** — the person/team accountable for it.
- **Last rotation** — date of last rotation, best-effort, or `unknown`.
- **Blast radius** — impact if leaked: `low` / `medium` / `high`.
- **Notes** — context, scope-to-confirm flags, cross-references.

**Legend**

- Blast radius: `high` = broad write / publish / signing / cross-system; `medium` =
  scoped service access; `low` = read-only, short-lived, or non-sensitive identifier.
- Status markers: `unknown` (no data yet), `TBD` (to be filled by owner), `confirm`
  (derived but needs verification in the source system).

**How it is organized.** Sections follow the secret classes from the proposal. The
GitHub Actions sections are the canonical list of GitHub-stored secrets (they answer the
"GitHub org / repo / environment secrets" classes directly). The type-based sections
(npm, cloud, mobile signing, third-party, SSH) cross-reference those CI secrets and add
the instances that live **outside** GitHub Actions, so nothing is double-counted.

**Data sources for the pre-filled rows.** GitHub Actions rows were derived from
`.github/workflows/**`; local-developer rows from [`.env.example`](../../.env.example),
as of the "Last reviewed" date. Scope (org vs repo vs environment) and human access
lists are **not** visible from the repository and are marked `confirm` / `TBD`.

---

## 1. GitHub Actions secrets — organization level

Org-level secrets are shared across repositories in the `tetherto` org. Whether a given
name below (section 2) is defined at org or repo scope is only visible in GitHub
Settings and must be confirmed there.

UI path: `GitHub > Organization (tetherto) > Settings > Secrets and variables > Actions`.

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| _TBD_ | Org Settings > Secrets and variables > Actions | Org admins (manage); selected repos/workflows (use) | DevOps pod | unknown | TBD | Enumerate org-scoped secrets here; some names in section 2 may actually be org-level (`confirm`). |

---

## 2. GitHub Actions secrets — repository level

Discovered from workflow usage in `.github/workflows/**`. Scope (org vs repo) needs
confirmation in Settings; listed here as the default case.

UI path: `GitHub > Repo (tetherto/qvac) > Settings > Secrets and variables > Actions`.

Common fields for all rows below unless noted: **Where it lives** =
`Repo Settings > Secrets and variables > Actions`; **Read/use access** = `org/repo
admins (manage); CI workflows on authorized/verified runs (use)`; **Owner of record** =
`DevOps pod (individual TBD)`; **Last rotation** = `unknown`.

### 2a. Authentication & publish tokens

| Identifier | Blast radius | Notes |
|---|---|---|
| `PAT_TOKEN` | high | GitHub Personal Access Token used for private-dep checkout and cross-repo automation across most workflows. Broad repo + `read:packages` scope; long-lived. Highest-priority rotation candidate. |
| `GITHUB_TOKEN` | low | Built-in, auto-issued per job and expires at job end. **Not a managed secret — excluded from rotation.** Listed for completeness. |
| `NPM_TOKEN` | high | npm registry token; can publish `@qvac` packages to npmjs (supply-chain impact). Also see [section 4](#4-npm-publish--registry-tokens). |
| `HF_TOKEN` | low | HuggingFace read token for model-license verification. Read-only. Also in [`.env.example`](../../.env.example). |

### 2b. Cloud / AWS (CI usage)

| Identifier | Blast radius | Notes |
|---|---|---|
| `AWS_OIDC_ROLE_ARN` | low | ARN of the role assumed via GitHub OIDC. Federated — **no static key stored.** Access governed by the role trust policy. See [section 5](#5-cloud-provider-keys-per-cloud-account). |
| `MODEL_S3_BUCKET` | low | S3 bucket name (identifier/config, not a credential). |
| `AWS_DEVICE_FARM_PROJECT_ARN_NMTCPP` | low | AWS Device Farm project ARN (identifier). |
| `AWS_DEVICE_FARM_PROJECT_ARN_OCR_FASTTEXT` | low | AWS Device Farm project ARN (identifier). |
| `AWS_DEVICE_FARM_PROJECT_ARN_TTS_GGML` | low | AWS Device Farm project ARN (identifier). |
| `AWS_DEVICE_FARM_PROJECT_ARN_ONNX_TTS` | low | AWS Device Farm project ARN (identifier). |
| `IOS_DEVICE_POOL_ARN_NMTCPP` | low | Device Farm device-pool ARN (identifier). |
| `IOS_DEVICE_POOL_ARN_OCR_FASTTEXT` | low | Device Farm device-pool ARN (identifier). |
| `IOS_DEVICE_POOL_ARN_TTS_GGML` | low | Device Farm device-pool ARN (identifier). |
| `IOS_DEVICE_POOL_ARN_ONNX_TTS` | low | Device Farm device-pool ARN (identifier). |
| `ANDROID_DEVICE_POOL_ARN_OCR_FASTTEXT` | low | Device Farm device-pool ARN (identifier). |

### 2c. Apple mobile signing (CI usage)

| Identifier | Blast radius | Notes |
|---|---|---|
| `TEST_APP_APPLE_DISTRIBUTION_CERTIFICATE` | high | Base64 Apple distribution certificate (p12). Account/team-wide signing identity. See [section 7](#7-mobile-signing-identities--certs--provisioning-profiles). |
| `APPLE_P12_PASSWORD` | high | Password protecting the p12 above. Useless apart, critical paired. |
| `TEST_APP_APPLE_PROVISIONING_PROFILE` | medium | Base64 provisioning profile (app-scoped). |
| `APPLE_KEYCHAIN_PASSWORD` | medium | Password for the temporary CI keychain. CI-local. |
| `APPLE_TEAM_ID` | low | Apple developer Team ID (identifier, not secret). |

### 2d. MQTT test-broker credentials (CI usage)

| Identifier | Blast radius | Notes |
|---|---|---|
| `MQTT_PASSWORD` | medium | Broker password used by SDK mobile tests. See [section 8](#8-third-party-service-tokens). |
| `MQTT_USERNAME` | medium | Broker username. |
| `MQTT_CA_CERT` | medium | Broker CA certificate. |
| `MQTT_HOST` | low | Broker host (connection config). |
| `MQTT_PORT` | low | Broker port (connection config). |
| `MQTT_PROTOCOL` | low | Broker protocol (connection config). |

### 2e. Third-party service tokens (CI usage)

| Identifier | Blast radius | Notes |
|---|---|---|
| `QASE_API_TOKEN` | medium | Qase test-management API token. See [section 8](#8-third-party-service-tokens). |
| `GOOGLE_SHEETS_CREDENTIALS` | medium | Google service-account credentials JSON (benchmark result export). |
| `GOOGLE_SHEET_ID` | low | Target spreadsheet ID (identifier). |

---

## 3. GitHub Actions secrets — environment level

Environment-scoped secrets gate jobs bound to a GitHub Environment. Two environments are
referenced by workflows: **`release`** and **`npm`**. Which secrets are bound to each
environment (and any required reviewers / wait timers) is only visible in Settings.

UI path: `GitHub > Repo (tetherto/qvac) > Settings > Environments > {release | npm}`.

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| _TBD_ | Settings > Environments > `release` | Environment reviewers (gate); jobs using `environment: release` | DevOps pod | unknown | TBD | List env-bound secrets + required reviewers (`confirm`). |
| _TBD_ | Settings > Environments > `npm` | Environment reviewers (gate); jobs using `environment: npm` | DevOps pod | unknown | TBD | Likely gates npm-publish credentials (`confirm`). |

---

## 4. npm publish / registry tokens

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| `NPM_TOKEN` (CI) | GitHub Actions (see [section 2a](#2a-authentication--publish-tokens)) | CI publish/read jobs | DevOps pod | unknown | high | Publishes `@qvac` packages to npmjs. |
| Automation / CI-bot npm token | 1Password (path `TBD`) / npmjs account | TBD | TBD | unknown | high | Scaffold — the source token mirrored into `NPM_TOKEN`, if separate. |
| Personal publish tokens | npmjs per-user settings | individual maintainers | each maintainer | unknown | high | Scaffold — enumerate any human-held publish tokens; revoke on departure. |
| GitHub Packages (GPR) auth | Uses `GITHUB_TOKEN` / `PAT_TOKEN` | CI | DevOps pod | n/a | medium | Dev-stream publishes use GPR via existing tokens; no separate token. |

---

## 5. Cloud provider keys (per cloud account)

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| AWS GitHub-OIDC role | AWS IAM (role); ARN in `AWS_OIDC_ROLE_ARN` | Assumed by CI via OIDC | DevOps pod | n/a | medium | Federated — no static key to rotate; manage via the role **trust policy**. Review trusted repos/branches. |
| AWS static IAM access keys | AWS IAM / 1Password (path `TBD`) | TBD | TBD | unknown | high | Scaffold — list any long-lived access-key pairs (humans or services) per account. |
| AWS root account credentials | 1Password (path `TBD`) | TBD (break-glass) | TBD | unknown | high | Scaffold — per AWS account; should be MFA-protected and rarely used. |
| Other cloud providers | 1Password (path `TBD`) | TBD | TBD | unknown | TBD | Scaffold — add a row per non-AWS cloud account if any (GCP, Azure, etc.). |

---

## 6. Self-hosted runner credentials

Self-hosted runners are used by GPU / Windows-NVIDIA / macOS workflows (e.g.
`integration-test-vla.yml`, `vulkaninfo.yml`, `win11-nvidia-image-builder.yml`,
benchmark workflows).

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| Runner registration / agent tokens | GitHub runner config / 1Password (path `TBD`) | TBD | DevOps pod | unknown | high | Scaffold — tokens that register runners to the org/repo. |
| Runner host SSH keys | Host `~/.ssh` / 1Password (path `TBD`) | TBD | DevOps pod | unknown | high | Scaffold — keys for administering runner hosts. |
| Runner host login / sudo creds | 1Password (path `TBD`) | TBD | DevOps pod | unknown | high | Scaffold — OS-level credentials for each self-hosted host. |

---

## 7. Mobile signing identities / certs / provisioning profiles

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| Apple distribution cert + p12 password (CI) | GitHub Actions (see [section 2c](#2c-apple-mobile-signing-ci-usage)) | CI iOS build jobs | DevOps / Mobile (TBD) | unknown | high | `TEST_APP_APPLE_DISTRIBUTION_CERTIFICATE` + `APPLE_P12_PASSWORD`. |
| Apple provisioning profile (CI) | GitHub Actions (see [section 2c](#2c-apple-mobile-signing-ci-usage)) | CI iOS build jobs | DevOps / Mobile (TBD) | unknown | medium | `TEST_APP_APPLE_PROVISIONING_PROFILE`. |
| Apple Developer account | Apple Developer portal / 1Password (path `TBD`) | TBD | TBD | unknown | high | Scaffold — account-level identity behind the certs above. |
| Google Play app-signing key | Play Console (Google-managed) / 1Password | TBD | TBD | unknown | high | Scaffold — if Android release signing exists. |
| Google Play upload key / keystore | 1Password (path `TBD`) | TBD | TBD | unknown | high | Scaffold — upload keystore + passwords. |

---

## 8. Third-party service tokens

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| `QASE_API_TOKEN` | GitHub Actions (see [section 2e](#2e-third-party-service-tokens-ci-usage)) | CI integration tests | DevOps / QA (TBD) | unknown | medium | Qase test-management API. |
| Google service account (`GOOGLE_SHEETS_CREDENTIALS`) | GitHub Actions (see [section 2e](#2e-third-party-service-tokens-ci-usage)) | CI benchmark export | DevOps pod | unknown | medium | Service-account JSON; scope to the target sheet. |
| MQTT broker credentials | GitHub Actions (see [section 2d](#2d-mqtt-test-broker-credentials-ci-usage)) | CI SDK mobile tests | DevOps pod | unknown | medium | `MQTT_USERNAME` / `MQTT_PASSWORD` / `MQTT_CA_CERT`. |
| HuggingFace token (`HF_TOKEN`) | GitHub Actions + [`.env.example`](../../.env.example) | CI + local dev | DevOps pod | unknown | low | Read-only model access. |
| Slack app / bot tokens | 1Password (path `TBD`) | TBD | TBD | unknown | medium | Scaffold — any Slack apps, webhooks, or bot tokens. |
| Monitoring / observability tokens | 1Password (path `TBD`) | TBD | TBD | unknown | medium | Scaffold — Sentry / Datadog / etc. if used. |

---

## 9. Human-held SSH keys on shared infra

| Identifier | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| Per-human SSH keys to shared hosts | Each user's `~/.ssh`; authorized_keys on hosts | individual engineers | each key owner | unknown | high | Scaffold — enumerate humans with SSH access to shared/build infra; primary offboarding revocation list. |
| Shared/role SSH keys | 1Password (path `TBD`) | TBD | DevOps pod | unknown | high | Scaffold — discourage; convert to per-human keys where possible. |

---

## 10. 1Password vaults

The human secret store. Enumerate the vaults and who can access each; deep secret rows
above should reference the relevant vault path.

| Identifier (vault) | Where it lives | Read/use access | Owner of record | Last rotation | Blast radius | Notes |
|---|---|---|---|---|---|---|
| _TBD vault name_ | 1Password | TBD (group membership) | TBD | n/a | TBD | Scaffold — one row per vault; note membership group and what classes of secret it holds. |

---

## 11. Local developer credentials (`.env.example`)

These are per-developer **local copies** of upstream credentials, not a separate class.
Each maps to a secret tracked above; rotation/revocation follows the upstream entry.

| Identifier | Where it lives | Maps to | Blast radius | Notes |
|---|---|---|---|---|
| `GH_TOKEN` | Developer `.env` (git-ignored) | GitHub PAT — cf. `PAT_TOKEN` ([2a](#2a-authentication--publish-tokens)) | high | Personal classic PAT (`repo`, `read:packages`). Per-developer; revoke on departure. |
| `HF_TOKEN` | Developer `.env` (git-ignored) | HuggingFace read token ([8](#8-third-party-service-tokens)) | low | Read-only. |
| `NPM_TOKEN` | Developer `.env` (git-ignored) | npm read token ([4](#4-npm-publish--registry-tokens)) | medium | Read-only for `@qvac` resolution per `.env.example`. |

---

## 12. Rotation log

Append one row each time a secret is rotated, per
[`secrets-rotation-policy.md`](secrets-rotation-policy.md#how-a-rotation-is-recorded).
For v1 this log is maintained manually.

| Date | Secret (identifier) | Class / section | Reason (cadence / trigger) | Rotated by | Approved by | New location confirmed | Notes |
|---|---|---|---|---|---|---|---|
| _none yet_ | | | | | | | First entry expected from the Q3 Tier-1 rotation proof. |
