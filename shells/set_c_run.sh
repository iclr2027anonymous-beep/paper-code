#!/usr/bin/env bash
# set_c_run.sh — Set C (mechanism dials: beta, eta, backbone, resolution),
# per the experiment protocol §Set C.
#
#   C-a  KL-weight dial: beta x eta over NoisyAG-med + medical_abstracts
#        (text) and CIFAR-10N aggre/worse + CIFAR-100N fine (image, frozen
#        RN50 EMBEDDING = the default image config; the backbone axis itself
#        belongs to C-b, the A-pair Swin cells are the cross-encoder
#        instrument at the fixed A dials).
#   C-b  embedding-regime dial: frozen RN50 / trainable RN50 / frozen Swin
#        x clean-set {2000, 3000, 5000} on CIFAR-10N worse + CIFAR-100N fine
#        (frozen RN50 == the Set A embedding mode; swin/trainable == pixel).
#   C-c  224px resolution probe: dopanim, frozen Swin, img 224, epochs
#        {10, 15} x beta {0.1, 0.5}, cs 2000 from the clean test pool.
#   C-c (cc_rn50)  the same dopanim stream behind frozen RN50 features at
#        224px, unsupervised (eta 0, cs 0), 5 epochs, beta 0.1.
#
# One GPU process at a time (serial). Every cell is idempotent: skipped when
# its json already exists under results/set_c/json; cells that coincide with
# a completed Set A run (beta 0.5 / eta 0.1 / cs 2000) are copied from
# results/set_a instead of re-run. Nothing runs until you launch it.
#
# Usage:  bash shells/set_c_run.sh [ca_text|ca_image|cb|cc|cc_rn50|all]
# Env overrides: SEEDS="42 43" BETAS="0.1 0.5" ETAS="0.5" CLEAN_SET=...
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
  ca_text|ca_image|cb|cc|cc_rn50|all) ;;
  *) echo "usage: $0 [ca_text|ca_image|cb|cc|cc_rn50|all]"; exit 2 ;;
esac

LOGDIR=/tmp/set_c_logs
OUTDIR="$PROJECT_ROOT/results/set_c"
AJSON="$PROJECT_ROOT/results/set_a/json"
mkdir -p "$LOGDIR" "$OUTDIR/json" "$OUTDIR/per_query"
RUN_LOG="$LOGDIR/set_c.log"
: > "$RUN_LOG"

# ── protocol constants (env-overridable) ───────────────────────────────
SEEDS="${SEEDS:-42 43 44 45 46}"
BETAS="${BETAS:-0.1 0.25 0.5 0.75 1.0}"
ETAS="${ETAS:-0.1 0.5}"
CLEAN_SET="${CLEAN_SET:-2000}"
VAL_SIZE="${VAL_SIZE:-2500}"
ROB_K="${ROB_K:-32}"
# EMIT_ONLY=1: re-emit protocol scores from saved checkpoints (no refit).
# Used to restore a score column on cells emitted without it; the skip guards
# above are lifted in that mode so existing cells are re-derived.
EMIT_ONLY="${EMIT_ONLY:-0}"
EMIT_FLAG=(); [ "$EMIT_ONLY" = 1 ] && EMIT_FLAG=(--emit-only)
EPOCHS_TEXT="${EPOCHS_TEXT:-15}"
EPOCHS_IMAGE="${EPOCHS_IMAGE:-10}"
DOPANIM_SIZE="${DOPANIM_SIZE:-224}"
BATCH_RN50="${BATCH_RN50:-256}"
BATCH_PIX="${BATCH_PIX:-64}"

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

# Copy a completed Set A cell (same dials: eta auto-resolved 0.1) instead of
# re-training it. $1 = the C tag; the A tag is the C tag without "_eta0.1".
reuse_a_cell() {
  local tag="$1" a_tag="${1/_eta0.1/}"
  if [ -f "$OUTDIR/json/$tag.json" ]; then
    return 0  # already present
  fi
  if [ -f "$AJSON/$a_tag.json" ]; then
    cp "$AJSON/$a_tag.json" "$OUTDIR/json/$tag.json"
    local anpz="$AJSON/../per_query/$a_tag.npz"
    [ -f "$anpz" ] && cp "$anpz" "$OUTDIR/per_query/$tag.npz"
    reuse=$((reuse+1)); echo "[reuse-A] $tag <- $a_tag" | tee -a "$RUN_LOG"
    return 0
  fi
  return 1
}

eval_slice_text() {
  case "$1" in
    noisyag_med) echo 2000 ;;
    *) echo 1000 ;;
  esac
}
eval_slice_image() {
  case "$1:$2" in
    cifar10n:aggre) echo 2500 ;;
    cifar10n:worse) echo 3000 ;;
    cifar100n:fine) echo 3000 ;;
    *) echo 1000 ;;
  esac
}

# ═══════════════════ C-a: beta x eta dial (text) ═══════════════════════════
run_ca_text_cell() { # ds noise nt seed beta eta
  local ds="$1" noise="$2" nt="$3" seed="$4" beta="$5" eta="$6"
  local ev; ev=$(eval_slice_text "$ds")
  local tag="text_${ds}_s${seed}_n${noise}_${nt}_e${EPOCHS_TEXT}_b${beta}_eta${eta}_cs${CLEAN_SET}"
  local json_dst="$OUTDIR/json/$tag.json" npz_dst="$OUTDIR/per_query/$tag.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  if [ "$EMIT_ONLY" != 1 ] && reuse_a_cell "$tag"; then return; fi
  total=$((total+1))
  echo "=== [C-a TEXT] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.text_lsnpc_sst2 \
    --dataset "$ds" --noise "$noise" --noise-type "$nt" \
    --epochs "$EPOCHS_TEXT" --beta "$beta" --eta "$eta" --seed "$seed" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/text_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
ca_text_phase() {
  local ds noise nt beta eta seed
  if [ -z "${C_DATASETS:-}" ] || [[ "$C_DATASETS" == *noisyag_med* ]]; then
    for beta in $BETAS; do for eta in $ETAS; do for seed in $SEEDS; do
      run_ca_text_cell noisyag_med 0.0 symmetric "$seed" "$beta" "$eta"
    done; done; done
  fi
  if [ -z "${C_DATASETS:-}" ] || [[ "$C_DATASETS" == *medical_abstracts* ]]; then
    for noise in 0.2 0.4; do
      for nt in symmetric idn; do
        for beta in $BETAS; do for eta in $ETAS; do for seed in $SEEDS; do
          run_ca_text_cell medical_abstracts "$noise" "$nt" "$seed" "$beta" "$eta"
        done; done; done
      done
    done
  fi
}

# ═══════════════════ C-a: beta x eta dial (image) ═══════════════════════════
run_ca_image_cell() { # ds view seed beta eta
  local ds="$1" view="$2" seed="$3" beta="$4" eta="$5"
  local ev; ev=$(eval_slice_image "$ds" "$view")
  local tag="image_${ds}_${view}_${view}_resnet50_s${seed}_e${EPOCHS_IMAGE}_b${beta}_eta${eta}_cs${CLEAN_SET}"
  local json_dst="$OUTDIR/json/$tag.json" npz_dst="$OUTDIR/per_query/$tag.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  if [ "$EMIT_ONLY" != 1 ] && reuse_a_cell "$tag"; then return; fi
  total=$((total+1))
  echo "=== [C-a IMAGE] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset "$ds" --noise-type "$view" \
    --epochs "$EPOCHS_IMAGE" --beta "$beta" --eta "$eta" --seed "$seed" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" \
    --backbone resnet50 --batch-size "$BATCH_RN50" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
ca_image_phase() {
  local view beta eta seed
  for view in aggre worse; do
    for beta in $BETAS; do for eta in $ETAS; do for seed in $SEEDS; do
      run_ca_image_cell cifar10n "$view" "$seed" "$beta" "$eta"
    done; done; done
  done
  for beta in $BETAS; do for eta in $ETAS; do for seed in $SEEDS; do
    run_ca_image_cell cifar100n fine "$seed" "$beta" "$eta"
  done; done; done
}

# ═══════════════════ C-b: backbone x clean-set (image) ══════════════════════
run_cb_image_cell() { # ds view bb seed cs
  local ds="$1" view="$2" bb="$3" seed="$4" cs="$5"
  local ev; ev=$(eval_slice_image "$ds" "$view")
  local tag="image_${ds}_${view}_${view}_${bb}_s${seed}_e${EPOCHS_IMAGE}_b0.5_eta0.1_cs${cs}"
  local json_dst="$OUTDIR/json/$tag.json" npz_dst="$OUTDIR/per_query/$tag.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  if [ "$bb" != resnet50train ]; then
    if [ "$EMIT_ONLY" != 1 ] && reuse_a_cell "$tag"; then return; fi
  fi
  local pix=() extra=(--backbone resnet50 --batch-size "$BATCH_RN50")
  case "$bb" in
    resnet50)     extra=(--backbone resnet50 --batch-size "$BATCH_RN50") ;;
    swin)         pix=(--pixel); extra=(--backbone swin --batch-size "$BATCH_PIX") ;;
    resnet50train) pix=(--pixel --trainable); extra=(--backbone resnet50 --batch-size "$BATCH_PIX") ;;
    *) echo "unknown bb $bb"; exit 2 ;;
  esac
  total=$((total+1))
  echo "=== [C-b IMAGE] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset "$ds" --noise-type "$view" \
    --epochs "$EPOCHS_IMAGE" --beta 0.5 --eta 0.1 --seed "$seed" \
    --clean-set-size "$cs" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" "${pix[@]}" "${extra[@]}" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
cb_image_phase() {
  local cs bb seed
  for cs in 2000 3000 5000; do
    for bb in resnet50 swin; do
      for seed in $SEEDS; do
        run_cb_image_cell cifar10n worse "$bb" "$seed" "$cs"
        run_cb_image_cell cifar100n fine "$bb" "$seed" "$cs"
      done
    done
  done
}

# ═══════════════════ C-c: dopanim 224px resolution probe ════════════════════
dopanim_prep_size() { # prints current data size (or 0 when missing)
  "$PY" - "$DOPANIM_SIZE" <<'PY' 2>/dev/null
import pickle, sys
from pathlib import Path
p = Path("data/dopanim/train_batch")
if not p.exists():
    print(0); sys.exit(0)
d = pickle.load(open(p, "rb"))
print(d["data"].shape[-1])
PY
}
run_cc_cell() { # seed epochs beta
  local seed="$1" epochs="$2" beta="$3"
  local tag="image_dopanim_swin_s${seed}_e${epochs}_b${beta}_eta0.1_cs${CLEAN_SET}_sz${DOPANIM_SIZE}"
  local json_dst="$OUTDIR/json/$tag.json" npz_dst="$OUTDIR/per_query/$tag.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  total=$((total+1))
  echo "=== [C-c DOPANIM] $tag (224px frozen Swin, cs=$CLEAN_SET from clean test pool) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset dopanim --noise-type symmetric \
    --epochs "$epochs" --beta "$beta" --eta 0.1 --seed "$seed" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice 1000 --rob-k "$ROB_K" \
    --pixel --backbone swin --batch-size "$BATCH_PIX" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  if [ -f "$json_dst" ]; then
    "$PY" - "$json_dst" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("transitions") or {}
wr = t.get("WR", 0); rw = t.get("RW", 0); net = wr - rw
verdict = ("GATEABLE" if (wr + rw) >= 200 and net >= 0 else
           "net-negative but edit stream exists" if (wr + rw) >= 200 else
           "ABSTAINING (no usable edits -> resolution-boundary null)")
print(f"    dopanim probe verdict: WR={wr} RW={rw} net={net} -> {verdict}",
      file=sys.stderr)
PY
  fi
  die_on_fail
}
ensure_dopanim_size() {
  local cur
  if [ "${DRY_RUN:-0}" = 1 ]; then cur="$DOPANIM_SIZE"; else cur=$(dopanim_prep_size); fi
  if [ "$cur" -ne "$DOPANIM_SIZE" ]; then
    if [ "${DO_PREP:-1}" != "1" ]; then
      echo "ERROR: dopanim pickles are ${cur}px but DOPANIM_SIZE=$DOPANIM_SIZE; "
           "set DO_PREP=1 to rebuild at ${DOPANIM_SIZE}px (or fix DOPANIM_SIZE)" \
        | tee -a "$RUN_LOG"
      exit 1
    fi
    echo "dopanim pickles are ${cur}px; rebuilding at ${DOPANIM_SIZE}px "
         "(scripts/dopanim_prep.py --size $DOPANIM_SIZE; rewrites "
         "data/dopanim/train_batch + test_batch from the extracted jpegs)" \
      | tee -a "$RUN_LOG"
    "$PY" scripts/dopanim_prep.py --size "$DOPANIM_SIZE" 2>&1 | tee -a "$RUN_LOG"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
      echo "ABORT: dopanim prep failed" | tee -a "$RUN_LOG"; exit 1
    fi
  fi
}
cc_phase() {
  local beta epochs seed
  ensure_dopanim_size
  for beta in 0.1 0.5; do
    for epochs in 10 15; do
      for seed in $SEEDS; do run_cc_cell "$seed" "$epochs" "$beta"; done
    done
  done
}

# C-c companion: the same dopanim stream behind the frozen ResNet-50 feature
# encoder (no --pixel), trained unsupervised (eta 0, no clean reference) for
# five epochs at beta 0.1, on a 3,000-example evaluation slice with one split
# permutation shared by all corrector seeds. The feature mode cannot use the
# full pool, so it trains on the loader's default train subset.
run_cc_rn50_cell() { # seed
  local seed="$1"
  local tag="image_dopanim_resnet50_s${seed}_e5_b0.1_eta0.0_cs0_sz${DOPANIM_SIZE}"
  local json_dst="$OUTDIR/json/$tag.json" npz_dst="$OUTDIR/per_query/$tag.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return; fi
  total=$((total+1))
  echo "=== [C-c DOPANIM RN50] $tag (224px frozen RN50 features, eta 0, cs 0) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset dopanim --noise-type symmetric \
    --epochs 5 --beta 0.1 --eta 0.0 --seed "$seed" \
    --split-seed 42 --allow-large-eval \
    --eval-slice 3000 --val-size "$VAL_SIZE" \
    --clean-set-size 0 --rob-k "$ROB_K" --batch-size 64 "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  local rc=$? run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" "$rc"
  else
    record "$tag" "$run_dir" "$json_dst" "$npz_dst" 1
  fi
  die_on_fail
}
cc_rn50_phase() {
  local seed
  ensure_dopanim_size
  for seed in $SEEDS; do run_cc_rn50_cell "$seed"; done
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
case "$MODE" in
  ca_text) ca_text_phase ;;
  ca_image) ca_image_phase ;;
  cb) cb_image_phase ;;
  cc) cc_phase ;;
  cc_rn50) cc_rn50_phase ;;
  all)
    ca_text_phase
    ca_image_phase
    cb_image_phase
    cc_phase
    cc_rn50_phase
    ;;
esac

echo "=== Set C ($MODE): ok=$ok skip=$skip reuse_A=$reuse fail=$fail new_runs=$total ===" | tee -a "$RUN_LOG"
if [ "$fail" -ne 0 ]; then
  echo "ERROR: $fail cell(s) failed — logs under $LOGDIR" | tee -a "$RUN_LOG"
  exit 1
fi
echo "results: $OUTDIR/json" | tee -a "$RUN_LOG"
