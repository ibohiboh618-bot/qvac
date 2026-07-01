#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QVAC-20557 — per-op verification (test-backend-ops) on the device.
#
# Runs ggml's standalone `test-backend-ops` on the phone to compare individual
# GPU ops (default MUL_MAT) against the CPU reference, and reports which op cases
# DIVERGE. This localizes "where exactly the GPU differs from CPU" WITHOUT the
# full model and WITHOUT a new build from the colleague — the operating agent can
# sweep ops autonomously.
#
#   results/<label>/backend-ops-<op>.log  — full test-backend-ops output
#   results/<label>/backend-ops-FAILS.txt — just the failing cases
#
# READ FIRST: test-backend-ops uses RANDOM inputs + a tolerance. For this bug it
# may PASS even though the real model miscomputes (a sibling bug did exactly that).
# So a FAIL here = a solid localization; a PASS does NOT clear the op — the real
# oracle is run-on-device.sh's dprobe_pw1_mulmat. See REFERENCE-what-to-test.md.
#
# Usage:
#   bash run-backend-ops.sh                       # MUL_MAT on Vulkan0
#   bash run-backend-ops.sh --op MUL_MAT          # a specific op
#   bash run-backend-ops.sh --all                 # all ops (slow)
#   bash run-backend-ops.sh --backend Vulkan0     # pick the backend (default Vulkan0)
#   bash run-backend-ops.sh --label probe2 --serial XXXX
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR/bundle"
RESULTS_ROOT="$SCRIPT_DIR/results"
DEVICE_DIR="/data/local/tmp/qvac-ttsg-verify"

OP="MUL_MAT"
BACKEND="Vulkan0"
ALL=0
LABEL=""
ADB_SERIAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --op)      OP="$2"; shift ;;
    --backend) BACKEND="$2"; shift ;;
    --all)     ALL=1 ;;
    --label)   LABEL="$2"; shift ;;
    --serial)  ADB_SERIAL="$2"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ADB=(adb); [ -n "$ADB_SERIAL" ] && ADB=(adb -s "$ADB_SERIAL")
say()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   OK %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   !! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

command -v adb >/dev/null 2>&1 || die "adb not found on PATH."
BIN=""
for c in "$BUNDLE_DIR/tools/test-backend-ops" "$BUNDLE_DIR/test-backend-ops"; do [ -f "$c" ] && BIN="$c"; done
[ -n "$BIN" ] || die "test-backend-ops not found in bundle (tools/). Ask the colleague to include it."
[ -f "$BUNDLE_DIR/libc++_shared.so" ] || die "bundle missing libc++_shared.so"

STATE="$("${ADB[@]}" get-state 2>/dev/null || true)"
[ "$STATE" = "device" ] || die "no authorized device (adb get-state='$STATE'). Cable / USB-debugging / 'Allow' — ask the human, don't loop."

[ -z "$LABEL" ] && { n=1; while [ -d "$RESULTS_ROOT/backend-ops$n" ]; do n=$((n+1)); done; LABEL="backend-ops$n"; }
OUTDIR="$RESULTS_ROOT/$LABEL"; mkdir -p "$OUTDIR"

say "staging test-backend-ops"
"${ADB[@]}" shell "mkdir -p $DEVICE_DIR" >/dev/null
"${ADB[@]}" push "$BIN" "$DEVICE_DIR/test-backend-ops" >/dev/null
"${ADB[@]}" shell "chmod 755 $DEVICE_DIR/test-backend-ops"
"${ADB[@]}" shell "[ -f $DEVICE_DIR/libc++_shared.so ]" >/dev/null 2>&1 || "${ADB[@]}" push "$BUNDLE_DIR/libc++_shared.so" "$DEVICE_DIR/libc++_shared.so" >/dev/null

OPSEL=""; [ "$ALL" = "0" ] && OPSEL="-o $OP"
LOG="$OUTDIR/backend-ops-$([ "$ALL" = "1" ] && echo all || echo "$OP").log"
say "running test-backend-ops ($([ "$ALL" = "1" ] && echo 'all ops' || echo "$OP"), backend $BACKEND)"
set +e
"${ADB[@]}" shell "LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/test-backend-ops test $OPSEL -b $BACKEND" > "$LOG" 2>&1
rc=$?
set -e

# strip ANSI colour so grep/agent parsing is clean
sed -i.bak 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null || true; rm -f "$LOG.bak"
grep -E 'FAIL|NMSE|not supported|error' "$LOG" > "$OUTDIR/backend-ops-FAILS.txt" 2>/dev/null || true

# grep -c prints "0" AND exits 1 on no-match, so use `|| true` (not `|| echo 0`,
# which would append a second line) and default empties to 0.
NPASS="$(grep -c ' OK$' "$LOG" 2>/dev/null || true)"; NPASS="${NPASS:-0}"
NFAIL="$(grep -cE 'FAIL$' "$LOG" 2>/dev/null || true)"; NFAIL="${NFAIL:-0}"
say "summary"
{
  echo "test-backend-ops — $([ "$ALL" = "1" ] && echo 'all ops' || echo "$OP") on $BACKEND — $(date)  (exit $rc)"
  echo "OK cases: $NPASS    FAIL cases: $NFAIL"
  echo
  if [ "$NFAIL" -gt 0 ]; then
    echo "FAILING op cases (decisive localizations):"
    grep -E 'FAIL$' "$LOG" | sed 's/^/  /'
  else
    echo "No FAILs. NOTE: test-backend-ops uses random inputs — a PASS does NOT clear the op for this"
    echo "data-specific bug. Trust run-on-device.sh's dprobe_pw1_mulmat as the real oracle."
  fi
} | tee "$OUTDIR/SUMMARY-backend-ops.txt"
echo
ok "collected into $OUTDIR/"
