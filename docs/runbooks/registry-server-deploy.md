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

## Node prerequisites (confirmed 2026-06-16)

The deploy assumes the following on every staging node; verified on
`registry-stg-01/02/03`. Re-check with
`packages/registry-server/scripts/deploy/node-diagnostics.sh` (run as the
run-as-user) if a node is rebuilt.

| Assumption | Confirmed value |
| --- | --- |
| Git checkout path | `/home/work/qvac` (remote `tetherto/qvac`, branch `main`) |
| Run-as-user | `work`, owns the checkout and the pm2 processes |
| Login → sudo | operator logs in as a metadata-key user, `sudo`s to `work` **passwordless** (`google-sudoers`) |
| pm2 process name | `registry` (fork mode), plus a separate `health-check` process (not touched) |
| Node toolchain | `node`/`npm`/`pm2` via **nvm** under `/home/work/.nvm` — `remote-update.sh` sources `nvm.sh` because CI runs non-interactively |
| Metrics | `http://127.0.0.1:9210/metrics` exposes `qvac_registry_is_indexer` |
| OS Login | **off** (project `enable-oslogin` unset) → metadata-key SSH path |
| Node version | ≥ 20 (currently v22.x) |
| Working tree | must be **clean** (tracked files); deploy fail-stops on drift |

> Zones differ per node (`registry-stg-01` europe-west6-c, `registry-stg-02`
> asia-southeast1-b, `registry-stg-03` us-central1-c), so zone is carried
> per-instance in `REGISTRY_STAGING_INSTANCES`.

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

**Identity vs. execution:** gcloud authenticates as the deploy service account.
**OS Login is off** on these nodes, so gcloud pushes an ephemeral SSH key to
instance metadata and the guest agent provisions the login user in the
`google-sudoers` group (passwordless sudo). The GCP-side audit trail (IAP tunnel
+ `compute.instances.setMetadata`) attributes the access to `registry-deploy@…`.
Once on the box, the session does `sudo -n su - <run-as-user>` (default `work`)
— **not** for identity, but because the git checkout, `node_modules`, and the
**per-user pm2 daemon** are owned by `work`, so the reload must run as that user.

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
| `GCP_PROJECT_ID` | project hosting the staging nodes (`tether-data-open-qvac`) |
| `REGISTRY_STAGING_INSTANCES` | JSON array of `{"name","zone"}` objects — nodes may be in **different zones**: `[{"name":"registry-stg-01","zone":"europe-west6-c"},{"name":"registry-stg-02","zone":"asia-southeast1-b"},{"name":"registry-stg-03","zone":"us-central1-c"}]` |
| `REGISTRY_NODE_REPO_PATH` | git checkout path on each node — `/home/work/qvac` |
| `REGISTRY_NODE_RUN_AS_USER` | optional; user that owns the checkout + pm2 process, sudo'd into after login (default `work`) |

> Repository variables (not environment variables) are required here so the
> un-environmented `resolve-deploy-targets` job can read the instance list.
> Repo-level vars are also visible to `deploy-staging` even though it scopes to
> the `release` environment.
>
> Per-instance zone lives in each `REGISTRY_STAGING_INSTANCES` entry (the nodes
> are spread across regions), so there is no single `GCP_ZONE` var. All nodes
> are assumed to share the same `REGISTRY_NODE_REPO_PATH`; if that diverges,
> promote it into the per-instance objects too.

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

**OS Login is off** on these nodes (confirmed — see Prerequisites), so the SA
reaches them via gcloud-managed **ephemeral SSH keys in instance metadata**. The
guest agent provisions the login user in the `google-sudoers` group, giving the
passwordless sudo the unattended `sudo -n su - work` depends on. Roles:

```bash
for role in \
  roles/iap.tunnelResourceAccessor \
  roles/compute.instanceAdmin.v1 \
  roles/compute.viewer ; do
  gcloud projects add-iam-policy-binding "<project>" \
    --member="serviceAccount:registry-deploy@<project>.iam.gserviceaccount.com" \
    --role="$role"
done
```

The nodes run as the **default compute service account**, and writing an
ephemeral SSH key to instance metadata requires "actAs" on that SA. Grant
`roles/iam.serviceAccountUser` on it (resource-level, not project-wide):

```bash
# find the VM's service account if unsure:
#   gcloud compute instances describe <node> --zone <zone> \
#     --format='value(serviceAccounts[0].email)'
gcloud iam service-accounts add-iam-policy-binding \
  <project-number>-compute@developer.gserviceaccount.com \
  --project="<project>" \
  --member="serviceAccount:registry-deploy@<project>.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

> Without this, `gcloud compute ssh` fails with *"The user does not have access
> to service account …-compute@developer.gserviceaccount.com … grant the
> iam.serviceAccountUser role."* (OS-Login-off metadata-key path only.)

> **Do not enable OS Login just for this** — turning it on makes GCE ignore
> metadata SSH keys, which can disrupt existing operator access. The nodes
> already use metadata-key SSH + `google-sudoers`.
>
> `roles/compute.instanceAdmin.v1` is broad. To tighten, replace it with a
> custom role granting only `compute.instances.setMetadata` and
> `compute.instances.get`, and/or grant it at the **instance** level on the
> three registry nodes rather than the whole project.
>
> If you later standardize on OS Login, switch this role to
> `roles/compute.osAdminLogin` (login + passwordless sudo) and enable OS Login
> on the nodes.

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
| SSH login refused / `Permission denied (publickey)` | SA lacks `compute.instances.setMetadata` (`instanceAdmin.v1`), or IAP firewall/role blocked |
| `Could not add SSH key to instance metadata … grant the iam.serviceAccountUser role` | SA lacks `roles/iam.serviceAccountUser` on the VMs' attached compute SA (step 4b) |
| `sudo: a password is required` | provisioned login user not in `google-sudoers` (unexpected on these nodes); add a `NOPASSWD` sudoers rule to the run-as-user |
| "tracked files are dirty on the node" | someone edited tracked files on the box; reconcile before deploy |
| health gate times out | process didn't rejoin as indexer; check `pm2 logs registry` (printed in the job output) |

## Rollback

The deploy checks out an exact commit. To roll a node back, re-run the workflow
via `workflow_dispatch` from the previous good commit, or break-glass:
`git checkout <previous-sha>` + `pm2 reload registry` on the node.
