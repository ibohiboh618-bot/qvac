#!/usr/bin/env bash
#
# node-diagnostics.sh — read-only preflight for a registry node.
#
# Run AS the run-as-user (e.g. `sudo su - work`) on a registry node to confirm
# the assumptions the automated deploy relies on (see
# docs/runbooks/registry-server-deploy.md "Node prerequisites"). Makes no
# changes. Optionally set REPO=/path/to/checkout (default: $HOME/qvac).
#
set +e

REPO="${REPO:-$HOME/qvac}"

echo "===== NODE: $(hostname) ====="
echo "-- identity --"; id; echo "HOME=$HOME"

echo "-- tools (run-as-user PATH) --"
for b in node npm git curl pm2; do
  if command -v "$b" >/dev/null 2>&1; then
    printf "  %-5s %-30s " "$b" "$(command -v "$b")"; "$b" --version 2>&1 | head -n1
  else
    echo "  $b: MISSING"
  fi
done

echo "-- pm2 processes --"
pm2 ls

echo "-- pm2 'registry' details --"
pm2 describe registry 2>/dev/null | grep -Ei "name|status|script path|exec cwd|script args" \
  || echo "  no pm2 process named 'registry' (see list above)"

echo "-- git checkouts under \$HOME (depth<=3) --"
find "$HOME" -maxdepth 3 -type d -name .git 2>/dev/null | while read -r g; do
  r="$(dirname "$g")"
  echo "  $r [branch:$(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null) sha:$(git -C "$r" rev-parse --short HEAD 2>/dev/null) remote:$(git -C "$r" remote get-url origin 2>/dev/null)]"
done

echo "-- registry checkout sanity ($REPO) --"
if [ -d "$REPO/.git" ]; then
  echo "  dirty(tracked): $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) file(s)"
  echo "  has package:    $([ -d "$REPO/packages/registry-server" ] && echo yes || echo NO)"
  echo "  has .env:       $([ -f "$REPO/packages/registry-server/.env" ] && echo yes || echo no)"
  echo "  licenses.json:  $([ -f "$REPO/packages/registry-server/data/licenses.json" ] && echo yes || echo no)"
else
  echo "  $REPO is not a git checkout — set REPO=<path> and re-run"
fi

echo "-- metrics endpoint (127.0.0.1:9210) --"
metrics_out="$(curl -fsS "http://127.0.0.1:9210/metrics" 2>/dev/null | grep -E '^qvac_registry_(is_indexer|model_count)')"
if [ -n "$metrics_out" ]; then
  echo "$metrics_out"
else
  echo "  no qvac_registry_* metrics on 9210"
fi

echo "-- node version (expect >= 20) --"; node -v 2>/dev/null
echo "===== END $(hostname) ====="
