#!/usr/bin/env bash
# set_d_run.sh — Set D (clean-budget axis, nested/fair), per
# the experiment protocol §Set D.
#
#   Claim C2 (embedding-regime mechanism): corrected-label quality is bounded
#   by the clean-reference budget only where the features can express the
#   correction. Budget fractions {5%, 20%, 100%} of the labelled reference
#   (default 2000 rows => clean sets {100, 400, 2000}) at fixed dials
#   beta 0.5 / eta 0.1, one seeded permutation per dataset with the clean set
#   as its PREFIX (nested: cs100 c cs400 c cs2000), val carved before the
#   prefix (invariant across budgets) — so any budget effect is causal.
#
#   TEXT  (epochs 15, frozen mpnet): NoisyAG-med (real crowd noise) +
#         medical_abstracts {0.2,0.4} x {sym,idn} (synthetic injection once
#         over the full label array, then subset by index).
#   IMAGE (epochs 5): CIFAR-10N worse + CIFAR-100N fine (real human labels)
#         x backbone {resnet50 frozen-EMBEDDING, resnet50train pixel} — the
#         frozen arm is the finding-#8 flat control, the trainable arm is
#         the C-b content (trainable restores cs-scaling or the image story
#         stays the rank gate).
#
# One GPU process at a time (serial). Every cell is idempotent: skipped when
# its json already exists under results/set_d/json; cells whose dials match a
# completed Set C / Set A run (cs 2000, eta 0.1, beta 0.5) are copied instead
# of re-run. Nothing runs until you launch it.
#
# Usage:  bash shells/set_d_run.sh [text|image|all]
# Env overrides:
#   SEEDS="42 43" BUDGET_REF=2000 BUDGET_FRACS="0.05 0.2 1.0"
#   BUDGET_SIZES="100 400 2000"   (explicit clean sizes override fracs)
#   TEXT_DATASETS="noisyag_med" IMAGE_DATASETS="cifar10n_worse"
#   IMAGE_BACKBONES="resnet50 resnet50train"
set -u
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ACCEL="${ACCEL:-accelerate}"
PY="${PY:-python}"

export CONDA_NO_PLUGINS=true
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4

MODE="${1:-all}"
case "$MODE" in
  text|image|all) ;;
  *) echo "usage: $0 [text|image|all]"; exit 2 ;;
esac

LOGDIR=/tmp/set_d_logs
OUTDIR="$PROJECT_ROOT/results/set_d"
CJSON="$PROJECT_ROOT/results/set_c/json"
AJSON="$PROJECT_ROOT/results/set_a/json"
mkdir -p "$LOGDIR" "$OUTDIR/json" "$OUTDIR/per_query"
RUN_LOG="$LOGDIR/set_d.log"
: > "$RUN_LOG"

# ── protocol constants (env-overridable) ───────────────────────────────
SEEDS="${SEEDS:-42 43 44 45 46}"
BUDGET_REF="${BUDGET_REF:-2000}"           # "labelled reference" = the 2k A/C labelled reference
BUDGET_FRACS="${BUDGET_FRACS:-0.05 0.2 1.0}"
BUDGET_SIZES="${BUDGET_SIZES:-}"
VAL_SIZE="${VAL_SIZE:-2500}"
ROB_K="${ROB_K:-32}"
EPOCHS_TEXT="${EPOCHS_TEXT:-15}"
EPOCHS_IMAGE="${EPOCHS_IMAGE:-5}"   # the reported image ladder; analyze_set_d_budget defaults to e5
BATCH_RN50="${BATCH_RN50:-256}"
BATCH_PIX="${BATCH_PIX:-64}"
BATCH_SWIN="${BATCH_SWIN:-64}"
TEXT_DATASETS="${TEXT_DATASETS:-}"
IMAGE_DATASETS="${IMAGE_DATASETS:-}"
IMAGE_BACKBONES="${IMAGE_BACKBONES:-resnet50}"

if [ -n "$BUDGET_SIZES" ]; then
  read -r -a BUDGETS <<< "$BUDGET_SIZES"
else
  budget_values=$("$PY" - "$BUDGET_REF" "$BUDGET_FRACS" <<'PYBUDGET'
import sys
reference = int(sys.argv[1])
values = [round(float(x) * reference) for x in sys.argv[2].split()]
if not values or any(v <= 0 for v in values) or len(values) != len(set(values)):
    raise SystemExit("Budgets must be positive and distinct.")
print(" ".join(map(str, values)))
PYBUDGET
) || exit 2
  read -r -a BUDGETS <<< "$budget_values"
fi
echo "Set D clean budgets: ${BUDGETS[*]} (reference $BUDGET_REF; fractions of the labelled reference: $BUDGET_FRACS)" | tee -a "$RUN_LOG"

total=0; ok=0; skip=0; fail=0; reuse=0
FAIL_MSG=""
record() { # tag run_dir json_dst npz_dst [rc]
  local tag="$1" run_dir="$2" json_dst="$3" npz_dst="$4" rc=${5:-0}
  if [ "$rc" -ne 0 ]; then
    fail=$((fail+1)); FAIL_MSG="cell exited with code $rc (log: $LOGDIR/$tag.log)"
    echo "  FAIL (exit=$rc) $tag — log: $LOGDIR/$tag.log" | tee -a "$RUN_LOG"
    return 1
  fi
  if [ ! -f "$run_dir/result.json" ] || ! grep -q 'cf_auroc' "$run_dir/result.json"; then
    fail=$((fail+1)); FAIL_MSG="result.json missing/invalid for $tag (log: $LOGDIR/$tag.log)"
    echo "  FAIL result-not-found $tag — log: $LOGDIR/$tag.log" | tee -a "$RUN_LOG"
    return 1
  fi
  cp "$run_dir/result.json" "$json_dst"
  [ -f "$run_dir"/per_query_scores_*.npz ] && cp "$run_dir"/per_query_scores_*.npz "$npz_dst"
  ok=$((ok+1)); echo "  OK -> $json_dst" | tee -a "$RUN_LOG"
}
die_on_fail() {
  if [ -n "$FAIL_MSG" ]; then
    echo "ABORT: first failure: $FAIL_MSG" | tee -a "$RUN_LOG"
    echo "Fix the cause, then re-run — completed cells are skipped (idempotent)." | tee -a "$RUN_LOG"
    exit 1
  fi
}

# Copy an already-completed C cell (same dials incl. eta0.1) or, stripping
# the _eta0.1 suffix, a completed Set A cell (frozen arms only). $1 = the D tag.
reuse_existing_cell() {
  local tag="$1" a_tag="${1/_eta0.1/}" src_dir src_tag
  if [ -f "$OUTDIR/json/$tag.json" ]; then return 0; fi
  for src_dir in "$CJSON" "$AJSON"; do
    src_tag="$tag"
    if [ "$src_dir" = "$AJSON" ]; then src_tag="$a_tag"; fi
    if [ -f "$src_dir/$src_tag.json" ]; then
      cp "$src_dir/$src_tag.json" "$OUTDIR/json/$tag.json"
      local anpz="${src_dir%/json}/per_query/$src_tag.npz"
      [ -f "$anpz" ] && cp "$anpz" "$OUTDIR/per_query/$tag.npz"
      reuse=$((reuse+1)); echo "[reuse] $tag <- $src_dir/$src_tag.json" | tee -a "$RUN_LOG"
      return 0
    fi
  done
  return 1
}

eval_slice_text() {
  case "$1" in
    noisyag_best)  echo 2500 ;;   # per-stream slices, as used for these streams in Set A
    noisyag_med)   echo 2000 ;;
    noisyag_worst) echo 3000 ;;
    *) echo 1000 ;;
  esac
}
eval_slice_image() {
  case "$1:$2" in
    cifar10n:worse) echo 3000 ;;
    cifar100n:fine) echo 3000 ;;
    *) echo 1000 ;;
  esac
}

# ══════════════════════ TEXT phase (epochs 15) ═════════════════════════════
run_text_cell() { # ds noise nt seed cs
  local ds="$1" noise="$2" nt="$3" seed="$4" cs="$5"
  local ev; ev=$(eval_slice_text "$ds")
  local tag="text_${ds}_s${seed}_n${noise}_${nt}_e${EPOCHS_TEXT}_b0.5_eta0.1_cs${cs}"
  local run_dir="/tmp/${tag}"
  local json_dst="$OUTDIR/json/${tag}.json"
  local npz_dst="$OUTDIR/per_query/${tag}.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  if reuse_existing_cell "$tag"; then return; fi
  total=$((total+1))
  echo "=== [D TEXT] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.text_lsnpc_sst2 \
    --dataset "$ds" --noise "$noise" --noise-type "$nt" \
    --epochs "$EPOCHS_TEXT" --beta 0.5 --eta 0.1 --seed "$seed" \
    --clean-set-size "$cs" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? found
  found=$(grep -o '\[out\] /tmp/text_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$found" ] && [ -d "$found" ]; then
    record "$tag" "$found" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
run_text_phase() {
  local ds noise nt seed cs
  for cs in "${BUDGETS[@]}"; do
    if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *noisyag_med* ]]; then
      for seed in $SEEDS; do run_text_cell noisyag_med 0.0 symmetric "$seed" "$cs"; done
    fi
    # The two further NoisyAG streams run ONLY when named in TEXT_DATASETS, so a
    # bare `set_d_run.sh text` keeps its original scope (noisyag_med + medical_abstracts).
    if [[ "$TEXT_DATASETS" == *noisyag_best* ]]; then
      for seed in $SEEDS; do run_text_cell noisyag_best 0.0 symmetric "$seed" "$cs"; done
    fi
    if [[ "$TEXT_DATASETS" == *noisyag_worst* ]]; then
      for seed in $SEEDS; do run_text_cell noisyag_worst 0.0 symmetric "$seed" "$cs"; done
    fi
    if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *medical_abstracts* ]]; then
      for noise in 0.2 0.4; do
        for nt in symmetric idn; do
          for seed in $SEEDS; do run_text_cell medical_abstracts "$noise" "$nt" "$seed" "$cs"; done
        done
      done
    fi
  done
}

# ══════════════════ IMAGE phase (epochs 5) ═════════════════════════════════
run_image_cell() { # ds view bb seed cs
  local ds="$1" view="$2" bb="$3" seed="$4" cs="$5"
  local ev; ev=$(eval_slice_image "$ds" "$view")
  local tag="image_${ds}_${view}_${view}_${bb}_s${seed}_e${EPOCHS_IMAGE}_b0.5_eta0.1_cs${cs}"
  local json_dst="$OUTDIR/json/${tag}.json"
  local npz_dst="$OUTDIR/per_query/${tag}.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  # Set A never ran the trainable pixel arm — skip the A fallback for it.
  if [ "$bb" != "resnet50train" ]; then
    if reuse_existing_cell "$tag"; then return; fi
  elif [ -f "$CJSON/$tag.json" ]; then
    cp "$CJSON/$tag.json" "$OUTDIR/json/$tag.json"
    local cn="${CJSON%/json}/per_query/$tag.npz"
    [ -f "$cn" ] && cp "$cn" "$npz_dst"
    reuse=$((reuse+1)); echo "[reuse] $tag <- C" | tee -a "$RUN_LOG"; return
  fi
  local pix=() extra=(--backbone resnet50 --batch-size "$BATCH_RN50")
  case "$bb" in
    resnet50)     extra=(--backbone resnet50 --batch-size "$BATCH_RN50") ;;
    resnet50train) pix=(--pixel --trainable); extra=(--backbone resnet50 --batch-size "$BATCH_PIX") ;;
    swin)         pix=(--pixel); extra=(--backbone swin --batch-size "$BATCH_SWIN") ;;
    *) echo "unknown backbone $bb"; exit 2 ;;
  esac
  total=$((total+1))
  echo "=== [D IMAGE] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset "$ds" --noise-type "$view" \
    --epochs "$EPOCHS_IMAGE" --beta 0.5 --eta 0.1 --seed "$seed" \
    --clean-set-size "$cs" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" "${pix[@]}" "${extra[@]}" \
    > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
run_image_phase() {
  local ds view bb seed cs
  for cs in "${BUDGETS[@]}"; do
    for bb in $IMAGE_BACKBONES; do
      if [ -z "$IMAGE_DATASETS" ] || [[ "$IMAGE_DATASETS" == *cifar10n* ]]; then
        for seed in $SEEDS; do run_image_cell cifar10n worse "$bb" "$seed" "$cs"; done
      fi
      if [ -z "$IMAGE_DATASETS" ] || [[ "$IMAGE_DATASETS" == *cifar100n* ]]; then
        for seed in $SEEDS; do run_image_cell cifar100n fine "$bb" "$seed" "$cs"; done
      fi
    done
  done
}


validate_seed_list() {
  "$PY" - "$1" <<'PYSEED'
import sys
values = [int(x) for x in sys.argv[1].split()]
if len(values) < 5 or len(values) != len(set(values)):
    raise SystemExit("Provide at least five distinct training seeds.")
PYSEED
}
validate_seed_list "$SEEDS" || exit 2

# ── Dispatch ───────────────────────────────────────────────────────────────
if [ "$MODE" = "text" ] || [ "$MODE" = "all" ]; then run_text_phase; fi
if [ "$MODE" = "image" ] || [ "$MODE" = "all" ]; then run_image_phase; fi

echo "=== Set D ($MODE): ok=$ok skip=$skip reuse=$reuse fail=$fail new_runs=$total ===" \
  | tee -a "$RUN_LOG"
if [ "$fail" -ne 0 ]; then
  echo "ERROR: $fail cell(s) failed — logs under $LOGDIR" | tee -a "$RUN_LOG"
  exit 1
fi
echo "results: $OUTDIR/json — run the budget analyzer:" | tee -a "$RUN_LOG"
echo "  $PY -m scripts.analyze_set_d_budget --json-dir $OUTDIR/json --out $OUTDIR" \
  | tee -a "$RUN_LOG"
