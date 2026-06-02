# Secrets rotation policy (v1)

Version: v1 draft &nbsp;|&nbsp; Last reviewed: 2026-06-02 &nbsp;|&nbsp; Owner: Yauheni
(`yauhenipankratovich-web`) &nbsp;|&nbsp; Security review: Giacomo (`GiacomoSorbiWork`)

The "when do we rotate it" half of the Q2 deliverable. It pairs with the
[`secrets-inventory.md`](secrets-inventory.md) catalog. See the [folder README](README.md)
for scope.

> **Scope.** This policy defines cadences, triggers, and recording for v1. It does **not**
> rotate any secret, select a central tool, or audit current compliance — those are Q3.

## Principles

1. **Every secret has an owner of record** (from the inventory). No owner = the DevOps
   pod lead owns it by default until reassigned.
2. **Rotate by blast radius first.** The higher the impact if leaked, the shorter the
   cadence.
3. **Triggers beat cadence.** A trigger (departure, suspected leak, scope change) forces
   immediate rotation regardless of how recently the secret was rotated.
4. **Record every rotation** so compliance can be measured later (Q4).
5. **Prefer eliminating long-lived secrets** (OIDC/federation, short-lived tokens) over
   rotating them.

## Cadence by blast radius

The default cadence is driven by the secret's blast-radius rating in the inventory.

| Blast radius | Rotate at least every | Rationale |
|---|---|---|
| `high` | **90 days** | Broad write / publish / signing / cross-system access. |
| `medium` | **180 days** | Scoped single-service access. |
| `low` | **365 days** | Read-only, short-lived, or non-sensitive identifier. |

Cadence is a **maximum age**, not a target. Rotating earlier is always compliant.

### Class to tier mapping (v1 defaults)

| Class / example | Default tier | Cadence |
|---|---|---|
| GitHub PAT (`PAT_TOKEN`), local `GH_TOKEN` | high | 90 days |
| npm publish tokens (`NPM_TOKEN`, automation/personal) | high | 90 days |
| Cloud static keys, AWS root credentials | high | 90 days |
| Apple distribution cert + p12 password | high | 90 days (and on expiry) |
| Self-hosted runner registration / host SSH / sudo | high | 90 days |
| Human SSH keys to shared infra | high | 90 days (and on departure) |
| Third-party service tokens (Qase, Google SA, monitoring, Slack) | medium | 180 days |
| MQTT broker credentials | medium | 180 days |
| Apple provisioning profile, CI keychain password | medium | 180 days |
| Read-only tokens (`HF_TOKEN`) | low | 365 days |
| Identifiers / config (bucket names, ARNs, `APPLE_TEAM_ID`, sheet IDs) | low | 365 days (review, not rotate) |

When the inventory and this table disagree, the **inventory's blast-radius rating wins**
and this mapping should be updated.

### Special cases (cadence does not apply as written)

- **`GITHUB_TOKEN`** — built-in, auto-issued per job and expires at job end. **Excluded
  from rotation.**
- **AWS GitHub-OIDC role (`AWS_OIDC_ROLE_ARN`)** — federated, no static key. Instead of
  rotating, **review the role trust policy** (trusted repos/branches) on the high-tier
  90-day cadence.
- **Signing certificates (Apple / Google)** — rotate on the high-tier cadence **or** on
  vendor expiry, whichever comes first, and always on a trigger.
- **Pure identifiers / config** (bucket names, ARNs, team IDs, sheet IDs) — not secrets;
  "rotation" means a yearly review that they are still correct, not regeneration.

## Mandatory rotation triggers

Any of these forces rotation **immediately**, ahead of cadence:

- **Departure / offboarding** — a person with read/use access leaves the team or company,
  or a contractor's engagement ends. Revoke + rotate everything they could read.
- **Suspected leak or exposure** — secret committed to git, pasted in logs/chat, leaked
  by a dependency, or any credible compromise signal.
- **Scope or ownership change** — a secret's permissions are widened, it is moved to a
  new system, or the owner of record changes.
- **Vendor / third-party compromise** — the upstream provider reports a breach affecting
  issued tokens.
- **Lost / stolen device** holding a secret (laptop, hardware token, signing machine).

On any trigger: rotate, confirm the new value works, revoke the old value, and add a
[rotation-log](#how-a-rotation-is-recorded) entry noting the trigger.

## How a rotation is recorded

Every rotation (cadence or trigger) is logged so compliance is auditable.

- **Where:** the **Rotation log** table in
  [`secrets-inventory.md`](secrets-inventory.md#12-rotation-log). For v1 this is updated
  manually via a normal PR.
- **Who performs it:** the secret's **owner of record** (or a DevOps pod member acting
  for them).
- **Who approves:** for `high` blast-radius secrets, **Security sign-off** (Giacomo,
  `GiacomoSorbiWork`) or the DevOps pod lead; `medium`/`low` may be self-approved by the
  owner.
- **What to record (log columns):** date, secret identifier, class/section, reason
  (cadence or which trigger), rotated by, approved by, "new location confirmed", notes.
- **Old value:** must be **revoked**, not just replaced, and the revocation noted.

## Measuring compliance (defined for Q4, not run in v1)

Compliance is **not audited in v1**. The metric is defined now so Q4 tooling can compute
it:

> **Rotation compliance %** = (secrets whose `last rotation` is within their tier's
> cadence) / (total secrets in scope), excluding `GITHUB_TOKEN` and pure identifiers.

This requires the inventory's `last rotation` fields to be populated — a Q3/Q4 follow-up
once owners backfill dates.

## What "good enough" looks like

### v1 (this quarter)

- Both documents exist and are linked from [`README.md`](README.md).
- Every secret **class** is listed with the captured fields.
- Each row has an **owner of record** and a **blast-radius** rating (best-effort).
- Cadences and triggers are **agreed** and signed off by Security.
- A **manual** rotation log exists and is referenced.
- It is acceptable that many `last rotation` values are `unknown` and that human access
  lists are `TBD` — v1 establishes the structure, not full data.

### Q4 maturity (target)

- Secrets managed in a **central tool** (Q3 selection + pilot) rather than scattered.
- **Automated expiry alerts** before cadence is breached.
- **Enforced rotation** (blocked publishes / access on overdue high-tier secrets).
- A **compliance dashboard** computing the metric above continuously.
- `last rotation` and access lists fully populated; offboarding has a one-click revoke
  list derived from the inventory.

## Sign-off & references

- **Owner of record:** Yauheni (`yauhenipankratovich-web`), DevOps pod.
- **Security review / sign-off:** Giacomo (`GiacomoSorbiWork`) — _pending_.

**References**

- DevOps Q2/Q3 proposal (off-repo) — Q2: "Secrets inventory + rotation policy doc v1 (no
  migration yet)"; gap callout: "Secrets, tokens, credentials (Group 3) — needs inventory
  + rotation policy + central tool decision".
- Q3 follow-ups that consume this policy (link when created):
  - "Central secrets tool: selection + pilot on one secret class".
  - "Tier-1 rotation cycle executed once (proof)" — first entry in the
    [Rotation log](secrets-inventory.md#12-rotation-log).
- [`docs/ci/TEAMS.md`](../ci/TEAMS.md) — team / pod ownership.
