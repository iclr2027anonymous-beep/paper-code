#!/usr/bin/env bash
# set_a_run.sh — Set A (regime curve) grid, per the experiment protocol.
#
#   Claim C1/C2: which per-query score informs correction validity, where.
#   Grid (beta 0.5, eta 0.1, clean-set 2000, rob-K 32):
#     TEXT (epochs 15):  NoisyAG-News best/med/worst (real, eval carve sized
#                        for the G4 mis-slice floor) + AG News {0.2,0.3,0.4} x
#                        {sym,idn} + SST-2 & medical_abstracts {0.2,0.4} x
#                        {sym,idn}; seeds 42-46.
#     IMAGE (epochs 10): CIFAR-10N aggre/worse + CIFAR-100N fine (real human)
#                        + EuroSAT {0.2,0.4} x {sym,idn} (synthetic control);
#                        frozen RN50 (embedding mode) AND frozen Swin (pixel
#                        mode) as separate per-dataset cells; seeds 42-46.
#                        IMAGE_BACKBONES filters the encoders (default both).
#   Splits per the experiment protocol SS0: eval carve default 1000, >=3000 on the
#   G4 headline settings (NoisyAG-worst, CIFAR-10N worse, CIFAR-100N fine);
#   validation >= 2500 rows on every per-query dataset.
#
# One GPU process at a time (serial); each cell is idempotent (skipped when its
# result json already exists). Nothing runs until you launch it.
#
# Usage:  bash shells/set_a_run.sh [text|image|all]
# Env overrides (examples):
#   SEEDS="42 43" BETA=0.3 ROB_K=4 \
#   TEXT_DATASETS="noisyag_worst medical_abstracts" \
#   bash shells/set_a_run.sh text
#
#   # EuroSAT sweep, split by encoder (paper rows use frozen RN50):
#   SEEDS="42 43 44 45 46 47 48 49" IMAGE_DATASETS="eurosat" \
#     IMAGE_BACKBONES="resnet50" bash shells/set_a_run.sh image
#   SEEDS="42 43 44" IMAGE_DATASETS="eurosat" \
#     IMAGE_BACKBONES="swin" bash shells/set_a_run.sh image
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

LOGDIR=/tmp/set_a_logs
OUTDIR="$PROJECT_ROOT/results/set_a"
mkdir -p "$LOGDIR" "$OUTDIR/json" "$OUTDIR/per_query"
RUN_LOG="$LOGDIR/set_a.log"
: > "$RUN_LOG"

# ── protocol constants (SS0; env-overridable) ─────────────────────────────
BETA="${BETA:-0.5}"
CLEAN_SET="${CLEAN_SET:-2000}"
VAL_SIZE="${VAL_SIZE:-2500}"
ROB_K="${ROB_K:-32}"
SEEDS="${SEEDS:-42 43 44 45 46}"
# EMIT_ONLY=1: re-emit protocol scores from saved ckpts (no refit), for a
# posterior-plausibility or pure re-emission pass.
EMIT_ONLY="${EMIT_ONLY:-0}"
EMIT_FLAG=(); [ "$EMIT_ONLY" = 1 ] && EMIT_FLAG=(--emit-only)
# EMIT_FRAMES=1: also record the label-free (self-referenced) variants of
# proximity and noise-robustness beside the clean-label values, in the same
# pass (experiments/deployed_frame.py). Kept under results/set_a/frames/ and
# results/set_a/frame_scores/. Combine with EMIT_ONLY=1 to add them to cells
# that already have checkpoints.
EMIT_FRAMES="${EMIT_FRAMES:-0}"
export EMIT_FRAMES
[ "$EMIT_FRAMES" = 1 ] && mkdir -p "$OUTDIR/frames" "$OUTDIR/frame_scores"
EPOCHS_TEXT="${EPOCHS_TEXT:-15}"
EPOCHS_IMAGE="${EPOCHS_IMAGE:-10}"
BATCH_RN50="${BATCH_RN50:-256}"
BATCH_SWIN="${BATCH_SWIN:-64}"
TEXT_DATASETS="${TEXT_DATASETS:-}"
IMAGE_DATASETS="${IMAGE_DATASETS:-}"
IMAGE_BACKBONES="${IMAGE_BACKBONES:-resnet50 swin}"

# G4 eval carves (mis-slice floor: >=1000 on headline settings, >=200 else).
eval_slice_text() {
  case "$1" in
    noisyag_best) echo 2500 ;;   # 9.8% x 2500 ≈ 245 mis > 200
    noisyag_med)  echo 2000 ;;   # 19.8% x 2000 ≈ 396 mis > 200
    noisyag_worst) echo 3000 ;;  # 37.8% x 3000 ≈ 1134 mis > 1000
    *) echo 1000 ;;
  esac
}
eval_slice_image() {
  case "$1:$2" in
    cifar10n:aggre)  echo 2500 ;;  # ~9% x 2500 ≈ 228 mis > 200
    cifar10n:worse)  echo 3000 ;;  # ~40% x 3000 ≈ 1200 mis > 1000
    cifar100n:fine)  echo 3000 ;;  # ~40% x 3000 ≈ 1200 mis > 1000
    eurosat:*)       echo 1500 ;;  # 20% x 1500 ≈ 300 mis > 200
    *) echo 1000 ;;
  esac
}

total=0; ok=0; fail=0; skip=0

FAIL_MSG=""   # first failure description; non-empty => sweep aborted

record() { # tag run_dir json_dst npz_dst [rc]
  local tag="$1" run_dir="$2" json_dst="$3" npz_dst="$4" rc=${5:-0}
  if [ "$rc" -ne 0 ]; then
    fail=$((fail+1))
    FAIL_MSG="cell exited with code $rc (log: $LOGDIR/$tag.log)"
    echo "  FAIL (exit=$rc) $tag — log: $LOGDIR/$tag.log" | tee -a "$RUN_LOG"
    return 1
  fi
  if [ ! -f "$run_dir/result.json" ] || ! grep -q 'cf_auroc' "$run_dir/result.json"; then
    fail=$((fail+1))
    FAIL_MSG="result.json missing/invalid for $tag (log: $LOGDIR/$tag.log)"
    echo "  FAIL result-not-found $tag — log: $LOGDIR/$tag.log" | tee -a "$RUN_LOG"
    return 1
  fi
  cp "$run_dir/result.json" "$json_dst"
  [ -f "$run_dir"/per_query_scores_*.npz ] && cp "$run_dir"/per_query_scores_*.npz "$npz_dst"
  if [ "$EMIT_FRAMES" = 1 ]; then
    if [ ! -s "$run_dir/frame_report.json" ]; then
      fail=$((fail+1))
      FAIL_MSG="EMIT_FRAMES=1 but no frame_report.json for $tag (log: $LOGDIR/$tag.log)"
      echo "  FAIL no-frames $tag — log: $LOGDIR/$tag.log" | tee -a "$RUN_LOG"
      return 1
    fi
    cp "$run_dir/frame_report.json" "$OUTDIR/frames/$tag.json"
    cp "$run_dir/frame_scores.npz" "$OUTDIR/frame_scores/$tag.npz"
  fi
  ok=$((ok+1)); echo "  OK -> $json_dst" | tee -a "$RUN_LOG"
}

die_on_fail() { # abort the sweep immediately on the first failed cell
  if [ -n "$FAIL_MSG" ]; then
    echo "ABORT: first failure: $FAIL_MSG" | tee -a "$RUN_LOG"
    echo "Fix the cause, then re-run — completed cells are skipped (idempotent)." | tee -a "$RUN_LOG"
    exit 1
  fi
}

# ── Text embeddings guard ─────────────────────────────────────────────────
check_embeddings() {
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  local missing=0
  for f in sst2/embeddings.pkl ag_news/train_emb.pkl \
           noisyag_news/embeddings.pkl medical_abstracts/embeddings.pkl; do
    if [ ! -f "data/$f" ]; then
      if [ "$f" = "medical_abstracts/embeddings.pkl" ]; then
        echo "medical_abstracts embeddings missing — running prep once" \
          | tee -a "$RUN_LOG"
        "$PY" scripts/medical_prep.py | tee -a "$RUN_LOG"
      fi
      if [ ! -f "data/$f" ]; then
        echo "MISSING data/$f" | tee -a "$RUN_LOG"; missing=1
      fi
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "ERROR: run the prep scripts first: scripts/text_sst2_prep.py, " \
         "scripts/noisyag_prep.py, scripts/medical_prep.py" | tee -a "$RUN_LOG"
    exit 1
  fi
}

# ══════════════════════ TEXT phase (epochs 15) ═════════════════════════════
run_text_cell() { # dataset noise noise_type seed
  local ds="$1" noise="$2" nt="$3" seed="$4"
  local ev; ev=$(eval_slice_text "$ds")
  local tag="text_${ds}_s${seed}_n${noise}_${nt}_e${EPOCHS_TEXT}_b${BETA}_cs${CLEAN_SET}"
  local run_dir="/tmp/${tag}"
  local json_dst="$OUTDIR/json/${tag}.json"
  local npz_dst="$OUTDIR/per_query/${tag}.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then
    echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return
  fi
  total=$((total+1))
  echo "=== [TEXT] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  $ACCEL launch --mixed_precision=bf16 -m scripts.text_lsnpc_sst2 \
    --dataset "$ds" --noise "$noise" --noise-type "$nt" \
    --epochs "$EPOCHS_TEXT" --beta "$BETA" --seed "$seed" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  record "$tag" "$run_dir" "$json_dst" "$npz_dst" $?
  die_on_fail
}

run_text_phase() {
  check_embeddings
  local ds nt noise seed
  # NoisyAG-News: real crowd labels, no injection; eval carved for G4.
  if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *noisyag_* ]]; then
    for ds in noisyag_best noisyag_med noisyag_worst; do
      for seed in $SEEDS; do run_text_cell "$ds" 0.0 symmetric "$seed"; done
    done
  fi
  # Synthetic text: noise injected once per full label array by the entry point.
  if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *ag_news* ]]; then
    for noise in 0.2 0.3 0.4; do
      for nt in symmetric idn; do
        for seed in $SEEDS; do run_text_cell ag_news "$noise" "$nt" "$seed"; done
      done
    done
  fi
  if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *sst2* ]]; then
    for noise in 0.2 0.4; do
      for nt in symmetric idn; do
        for seed in $SEEDS; do run_text_cell sst2 "$noise" "$nt" "$seed"; done
      done
    done
  fi
  if [ -z "$TEXT_DATASETS" ] || [[ "$TEXT_DATASETS" == *medical_abstracts* ]]; then
    for noise in 0.2 0.4; do
      for nt in symmetric idn; do
        for seed in $SEEDS; do run_text_cell medical_abstracts "$noise" "$nt" "$seed"; done
      done
    done
  fi
}

# ══════════════════════ IMAGE phase (epochs 10) ═══════════════════════════
run_image_cell() { # dataset view_or_noise noise_type backbone seed
  local ds="$1" view="$2" nt="$3" backbone="$4" seed="$5"
  local ev; ev=$(eval_slice_image "$ds" "$view")
  local pix=() extra=()
  if [ "$backbone" = "swin" ]; then
    pix=(--pixel); extra=(--batch-size "$BATCH_SWIN")
  else
    extra=(--batch-size "$BATCH_RN50")
  fi
  local tag="image_${ds}_${view}_${nt}_${backbone}_s${seed}_e${EPOCHS_IMAGE}_b${BETA}_cs${CLEAN_SET}"
  local json_dst="$OUTDIR/json/${tag}.json"
  local npz_dst="$OUTDIR/per_query/${tag}.npz"
  if [ "${DRY_RUN:-0}" = 1 ]; then printf "CELL %s\n" "$tag"; total=$((total+1)); return 0; fi
  if [ "$EMIT_ONLY" != 1 ] && [ -f "$json_dst" ]; then
    echo "[skip] $tag" | tee -a "$RUN_LOG"; skip=$((skip+1)); return
  fi
  total=$((total+1))
  echo "=== [IMAGE] $tag (eval=$ev val=$VAL_SIZE rob-K=$ROB_K) ===" | tee -a "$RUN_LOG"
  # Real-noise datasets: noise is fixed at 0 (view selects the human label
  # set). EuroSAT: view encodes the rate, nt the injection type.
  local noise_arg nt_arg
  if [[ "$ds" == cifar10n || "$ds" == cifar100n ]]; then
    noise_arg=0.0; nt_arg="$view"
  else
    noise_arg="$view"; nt_arg="$nt"
  fi
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset "$ds" --noise "$noise_arg" --noise-type "$nt_arg" \
    --epochs "$EPOCHS_IMAGE" --beta "$BETA" --seed "$seed" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" \
    --backbone "$backbone" "${pix[@]}" "${extra[@]}" "${EMIT_FLAG[@]}" > "$LOGDIR/$tag.log" 2>&1
  # Honour the real exit code (stale /tmp result.json from a previous run
  # must not be recorded as new when the cell crashes or misses its bundle).
  local rc=$?
  local run_dir
  run_dir=$(grep -o '\[out\] /tmp/image_[^ ]*' "$LOGDIR/$tag.log" | head -1 | sed 's#^\[out\] ##')
  record "$tag" "$run_dir" "$json_dst" "$npz_dst" $rc
  die_on_fail
}

run_image_phase() {
  local backbone view nt noise seed ds
  # Real-image settings: CIFAR-10N aggre/worse, CIFAR-100N fine (both backbones).
  if [ -z "$IMAGE_DATASETS" ] || [[ "$IMAGE_DATASETS" == *cifar10n* ]]; then
    for view in aggre worse; do
      for backbone in $IMAGE_BACKBONES; do
        for seed in $SEEDS; do run_image_cell cifar10n "$view" "$view" "$backbone" "$seed"; done
      done
    done
  fi
  if [ -z "$IMAGE_DATASETS" ] || [[ "$IMAGE_DATASETS" == *cifar100n* ]]; then
    for backbone in $IMAGE_BACKBONES; do
      for seed in $SEEDS; do run_image_cell cifar100n fine fine "$backbone" "$seed"; done
    done
  fi
  # Synthetic image control: EuroSAT (clean benchmark, injected noise).
  if [ -z "$IMAGE_DATASETS" ] || [[ "$IMAGE_DATASETS" == *eurosat* ]]; then
    for noise in 0.2 0.4; do
      for nt in symmetric idn; do
        for backbone in $IMAGE_BACKBONES; do
          for seed in $SEEDS; do run_image_cell eurosat "$noise" "$nt" "$backbone" "$seed"; done
        done
      done
    done
  fi
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

echo "=== Set A ($MODE): ok=$ok skip=$skip fail=$fail new_runs=$total ===" | tee -a "$RUN_LOG"
if [ "$fail" -ne 0 ]; then
  echo "ERROR: $fail cell(s) failed — logs under $LOGDIR" | tee -a "$RUN_LOG"
  exit 1
fi
echo "results: $OUTDIR/json" | tee -a "$RUN_LOG"
