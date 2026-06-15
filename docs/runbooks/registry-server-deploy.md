# Runbook — Automated registry-server staging deploy

How CI deploys the `registry-server` package to the staging registry nodes
before the model `sync-staging` step runs, and the one-time GCP + GitHub setup
required to enable it.

- Ticket: QVAC-19278
- Workflow: `.github/workflows/pr-models-validation-registry-server.yml`
- Composite action: `.github/actions/deploy-registry-server/action.yml`
- Remote script: `packages/registry-server/scripts/deploy/remote-update.sh`
- Nodes (Tier-1, GCP): `registry-stg-01/02/03` (see `docs/devops/TIER-1-SCOPE.md` §C, inventory ref A3)

## Why this exists

`scripts/sync-models.js` runs in CI and sends `add-model` RPCs (carrying a
`licenseId`) to a registry indexer. On the node, `addModel()` →
`_ensureLicense()` reads `data/licenses.json` and
`data/licenses/<spdxId>/LICENSE.txt` **from local disk**. If a model PR landed
before the node was updated with the new license files, the RPC fails with
`License <id> not available` (the PR #2187 incident). The deploy job updates all
nodes first, so whichever indexer the RPC reaches already knows the license.

## Flow

```
detect-changes ─▶ validate-json ─┐
                 test ───────────┤
                                 ├─▶ resolve-deploy-targets ─▶ deploy-staging ─▶ sync-staging ─▶ smoke-test
label-gate ──────────────────────┘     (reads instance list)   (1 node at a time)   (gated on deploy success)
```

- `deploy-staging` rolls nodes **one at a time** (`max-parallel: 1`,
  `fail-fast: true`). Each node must report `qvac_registry_is_indexer 1` on its
  local `/metrics` endpoint before the next node is touched — this preserves
  Autobase quorum and write availability.
- `sync-staging` only runs when `deploy-staging` succeeded. If any node fails to
  deploy or rejoin as indexer, sync does not run.
- Deploy logs appear in the Actions UI per node (`gcloud compute ssh` output is
  streamed); no SSH into the box is needed to diagnose a failure.

## Authentication model (keyless)

CI authenticates to GCP via **Workload Identity Federation (OIDC)** — no
long-lived SSH key or service-account JSON is stored. SSH to the nodes goes
**through IAP** (`gcloud compute ssh --tunnel-through-iap`), which manages
ephemeral SSH keys and requires no inbound public SSH.

The deploy job runs on the self-hosted `qvac-ubuntu2204-x64` runner. The runner
only needs **egress** to Google APIs (`oauth2.googleapis.com`,
`sts.googleapis.com`, `iap.googleapis.com`, `compute.googleapis.com`). SSH
**ingress** to the nodes comes from Google's IAP range, not the runner's IP.

## GitHub configuration

Reference names only — never inline values. **Scope matters:** the
`resolve-deploy-targets` job does not declare an environment, so anything it
reads (`REGISTRY_STAGING_INSTANCES`) MUST be a **repository variable** —
environment-scoped variables are invisible to jobs that don't reference the
environment. The two credentials stay as **`release` environment secrets**
(Tier-1, reviewer-gated); `deploy-staging` reads them and declares
`environment: release`.

### `release` environment secrets

| Name | Example / description |
| --- | --- |
| `GCP_WIF_PROVIDER` | `projects/<num>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | `registry-deploy@<project>.iam.gserviceaccount.com` |

### Repository variables (non-secret; visible to all jobs)

| Name | Example / description |
| --- | --- |
| `GCP_PROJECT_ID` | project hosting the staging nodes |
| `GCP_ZONE` | zone of the staging nodes, e.g. `europe-west6-a` |
| `REGISTRY_STAGING_INSTANCES` | JSON array, e.g. `["registry-stg-01","registry-stg-02","registry-stg-03"]` |
| `REGISTRY_NODE_REPO_PATH` | absolute path of the git checkout on each node, e.g. `/opt/qvac` |

> Repository variables (not environment variables) are required here so the
> un-environmented `resolve-deploy-targets` job can read the instance list.
> Repo-level vars are also visible to `deploy-staging` even though it scopes to
> the `release` environment.
>
> All three nodes are assumed to share the same zone and repo path. If that ever
> diverges, promote `GCP_ZONE` / `REGISTRY_NODE_REPO_PATH` into the
> `REGISTRY_STAGING_INSTANCES` entries and resolve them per-instance in
> `resolve-deploy-targets`.

## One-time GCP setup (scaffold)

Replace placeholders (`<...>`). Run with an account that can manage IAM, WIF, and
the nodes. These commands are idempotent enough to re-run, but review before
applying — creating IAM bindings is a state change.

### 1. Service account

```bash
gcloud iam service-accounts create registry-deploy \
  --project="<project>" \
  --display-name="registry-server CI deploy"
```

### 2. Workload Identity Federation pool + GitHub provider

```bash
gcloud iam workload-identity-pools create github-actions \
  --project="<project>" --location="global" \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc qvac \
  --project="<project>" --location="global" \
  --workload-identity-pool="github-actions" \
  --display-name="tetherto/qvac" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository == 'tetherto/qvac'"
```

### 3. Let only this repo (and ideally only `main`/dispatch) impersonate the SA

```bash
PROJECT_NUMBER="$(gcloud projects describe "<project>" --format='value(projectNumber)')"

gcloud iam service-accounts add-iam-policy-binding \
  "registry-deploy@<project>.iam.gserviceaccount.com" \
  --project="<project>" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/attribute.repository/tetherto/qvac"
```

`GCP_WIF_PROVIDER` is then:
`projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/providers/qvac`

### 4. IAM roles for the deploy SA

Least-privilege set for IAP SSH via OS Login:

```bash
for role in \
  roles/iap.tunnelResourceAccessor \
  roles/compute.osLogin \
  roles/compute.viewer ; do
  gcloud projects add-iam-policy-binding "<project>" \
    --member="serviceAccount:registry-deploy@<project>.iam.gserviceaccount.com" \
    --role="$role"
done
```

Enable OS Login on the nodes (project- or instance-level):

```bash
gcloud compute project-info add-metadata \
  --project="<project>" \
  --metadata enable-oslogin=TRUE
```

> If OS Login is not used, the SA instead needs
> `roles/compute.instanceAdmin.v1` so `gcloud compute ssh` can publish an
> ephemeral key to instance metadata. OS Login is preferred.

### 5. Firewall — allow IAP to reach SSH

```bash
gcloud compute firewall-rules create allow-iap-ssh \
  --project="<project>" \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=registry-node
```

Tag the nodes (`--tags registry-node`) or scope the rule to their network as
appropriate. `35.235.240.0/20` is Google's published IAP forwarding range.

## Operating the pipeline

- **Normal:** merge a models PR to `main` → workflow runs → all nodes deploy →
  sync runs → smoke test. Nothing manual.
- **Dry-run (validate plumbing without touching nodes):** Actions →
  *PR Validation and Staging Sync (Registry-server)* → **Run workflow** →
  enable **deploy_dry_run**. This exercises WIF auth + IAP connectivity + the
  remote script's preflight, performs **no** checkout/reload, and **skips**
  `sync-staging`.
- **Break-glass (manual):** the manual steps in
  `packages/registry-server/docs/DEPLOYMENT_GUIDE.md` still work — SSH to the
  node, `git pull`, `pm2 reload registry`.

## Failure triage

| Symptom | Likely cause |
| --- | --- |
| `resolve-deploy-targets` fails with "not set" | `vars.REGISTRY_STAGING_INSTANCES` missing/empty |
| auth step fails | WIF provider/SA wrong, or repo not bound (step 3) |
| `gcloud compute ssh` times out | IAP firewall (step 5) or `roles/iap.tunnelResourceAccessor` missing |
| "tracked files are dirty on the node" | someone edited tracked files on the box; reconcile before deploy |
| health gate times out | process didn't rejoin as indexer; check `pm2 logs registry` (printed in the job output) |

## Rollback

The deploy checks out an exact commit. To roll a node back, re-run the workflow
via `workflow_dispatch` from the previous good commit, or break-glass:
`git checkout <previous-sha>` + `pm2 reload registry` on the node.
