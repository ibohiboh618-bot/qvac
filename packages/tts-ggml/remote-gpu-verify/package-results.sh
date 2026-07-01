#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QVAC-20557 — package all collected iterations into one zip to send back.
#
# Bundles results/ (every iterN/ folder + their SUMMARY/diag/result files) plus
# an optional FINDINGS.md the operating agent should write, into a single
# qvac-mali-verify-<timestamp>.zip. No device interaction; safe to run anytime.
#
# Usage: bash package-results.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$SCRIPT_DIR/results"

[ -d "$RESULTS_ROOT" ] || { echo "no results/ yet — run run-on-device.sh first" >&2; exit 1; }

TS="$(date +%Y%m%d-%H%M%S)"
ZIP="$SCRIPT_DIR/qvac-mali-verify-$TS.zip"

# A top-level index so the requester sees what was collected at a glance.
{
  echo "QVAC-20557 Mali-Vulkan verification — collected $TS"
  echo
  echo "Iterations:"
  for d in "$RESULTS_ROOT"/*/; do
    [ -d "$d" ] || continue
    echo "  - $(basename "$d")"
    sed 's/^/        /' "$d/SUMMARY.txt" 2>/dev/null | sed -n '1,8p' || true
  done
  echo
  echo "If present, FINDINGS.md (written by the operating agent) has the analysis + verdict."
} > "$RESULTS_ROOT/INDEX.txt"

if command -v zip >/dev/null 2>&1; then
  ( cd "$SCRIPT_DIR" && zip -rq "$ZIP" results $( [ -f FINDINGS.md ] && echo FINDINGS.md ) )
else
  # zip not installed → fall back to tar.gz
  ZIP="${ZIP%.zip}.tgz"
  ( cd "$SCRIPT_DIR" && tar czf "$ZIP" results $( [ -f FINDINGS.md ] && echo FINDINGS.md ) )
fi

echo "wrote $ZIP"
echo "Send this file back to the requester."
