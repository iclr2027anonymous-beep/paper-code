#!/usr/bin/env bash
# set_b_run.sh — Set B (selective correction / operational gate), per
# the experiment protocol §Set B.
#
#   Claim C3 post-hoc + gate hierarchy of C2. Pure OFFLINE analysis over the
#   per-query artifacts of the Set A cells (results/set_a/json + per_query):
#   for every per-query-capable A cell it computes the gate operating
#   characteristic (full keep-curve gated label accuracy vs coverage,
#   deltas vs blanket/never-correct with paired bootstrap CIs, random
#   rejection control at R resamples, damage-removed vs repair-retained,
#   keep-curve AUC, bottom-decile broke rate) for the single axes
#   {1-f*, r, p*, H, pi} and the conjunctions {(1-f*)&r, (1-f*)&r&pi-floor}.
#
# No GPU and no training: CPU numpy only. Nothing runs until you launch it.
#
# Usage:  bash shells/set_b_run.sh [text|image|all]
# Env overrides:
#   SET_A_JSON=... SET_A_PER_QUERY=... OUTDIR=...  (artifact locations)
#   MODALITY=text|image  R_RESAMPLE=200  N_BOOT=1000  (see the analyzer)
#
# The same gate battery runs over the Set C cells by pointing the dirs at the
# Set C archive, e.g.:
#   SET_A_JSON=results/set_c/json SET_A_PER_QUERY=results/set_c/per_query \
#   OUTDIR=results/set_c/gates bash shells/set_b_run.sh
set -u
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python}"

MODE="${1:-all}"
case "$MODE" in
  text|image|all) ;;
  *) echo "usage: $0 [text|image|all]"; exit 2 ;;
esac

LOG=/tmp/set_b.log
: > "$LOG"

echo "=== Set B gate analysis ($MODE) ===" | tee -a "$LOG"

if [ "$MODE" = all ] && [ -n "${MODALITY:-}" ]; then
  MODE="$MODALITY"
fi
ARGS=()
if [ "$MODE" != all ]; then ARGS+=(--modality "$MODE"); fi
if [ -n "${MAX_CELLS:-}" ]; then ARGS+=(--max-cells "$MAX_CELLS"); fi
if [ -n "${R_RESAMPLE:-}" ]; then ARGS+=(--r-resample "$R_RESAMPLE"); fi
if [ -n "${N_BOOT:-}" ]; then ARGS+=(--n-boot "$N_BOOT"); fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "Set B reads Set A's five-seed JSON/NPZ cells; no training is scheduled."
  exit 0
fi
"$PY" -m scripts.check_a_inputs \
  --json-dir "${SET_A_JSON:-results/set_a/json}" \
  --npz-dir "${SET_A_PER_QUERY:-results/set_a/per_query}" \
  --seeds "${SEEDS:-42 43 44 45 46}" --modality "$MODE" || exit 1

"$PY" -m scripts.analyze_set_b_gates \
  --json-dir "${SET_A_JSON:-results/set_a/json}" \
  --npz-dir  "${SET_A_PER_QUERY:-results/set_a/per_query}" \
  --out      "${OUTDIR:-results/set_b}" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

echo "=== Set B ($MODE) exit=$rc — curves: ${OUTDIR:-results/set_b}/keep_curves.csv, summary: ${OUTDIR:-results/set_b}/summary.json ===" \
  | tee -a "$LOG"
exit $rc
