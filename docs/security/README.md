# Secrets management (v1)

This folder is the Q2 deliverable for the **secrets / tokens / credentials** strategic
gap (Group 3) called out in the DevOps Q2/Q3 proposal. It contains two documents:

- [`secrets-inventory.md`](secrets-inventory.md) — a v1 inventory of every secret class
  the team owns, with where each lives, who can use it, who owns it, last rotation
  (best-effort), and blast radius if leaked.
- [`secrets-rotation-policy.md`](secrets-rotation-policy.md) — rotation policy v1:
  cadence per class, mandatory rotation triggers, and how a rotation is recorded.

## Why this exists

Today secrets are spread across 1Password vaults, GitHub Actions secrets (org / repo /
environment), npm tokens, cloud provider keys, mobile signing identities, self-hosted
runner credentials, third-party service tokens, and human SSH keys — with **no single
list and no agreed rotation cadence**. That makes it impossible to:

- say what to revoke when someone leaves,
- measure rotation compliance, or
- plan the Q3 migration to a central secrets tool.

These two docs close the "what do we have / when do we rotate it" gap. They do **not**
rotate anything or pick a tool — see [Out of scope](#out-of-scope-q3).

## Source of truth

> These documents are a **catalog and a policy**, not a secret store. They contain
> **identifiers and metadata only — never secret values**. The live values and access
> lists are managed in their respective systems (GitHub at
> <https://github.com/orgs/tetherto>, 1Password, the relevant cloud consoles, Apple
> Developer / Google Play). Where a field cannot be derived from this repository it is
> marked `unknown` or `TBD` and must be completed by the owner of record.

The repo-discoverable rows in the inventory were derived from `.github/workflows/**`
and [`.env.example`](../../.env.example). Everything else is scaffolded for the owning
team to fill in.

## Scope (v1)

In scope for Q2:

- Inventory of every secret class with the captured fields.
- A written rotation policy: cadences, triggers, and recording method.
- Security sign-off (see below).

## Out of scope (Q3)

- Actually rotating any secret.
- Selecting or piloting a central secrets tool.
- Auditing whether existing secrets currently comply with the policy.

## Inputs to Q3

Both Q3 tickets take these documents as their input:

- **Central secrets tool: selection + pilot on one secret class** — uses the inventory's
  class list and blast-radius ratings to scope the pilot.
- **Tier-1 rotation cycle executed once (proof)** — uses the policy's cadence/trigger
  rules and the Rotation log table in the inventory to record the proof rotation.

When those tickets are created, link them here so the trail is two-way.

## Ownership and sign-off

- **Owner of record (this deliverable):** Yauheni (`yauhenipankratovich-web`), DevOps pod.
- **Security review / sign-off:** Giacomo (`GiacomoSorbiWork`).
- **Maintaining team:** DevOps pod (see [`docs/ci/TEAMS.md`](../ci/TEAMS.md) and
  [`.github/teams/devops.json`](../../.github/teams/devops.json)).
