#!/usr/bin/env bash
# set_e_run.sh — Set E (end-to-end: downstream retrain on corrected streams),
# per the experiment protocol §Set E.
#
#   C3 end-to-end: does a model trained from scratch on score-selected
#   corrected labels generalise better on the clean test than the same model
#   on the original labels? Pipeline per testbed:
#
#     1. fit     — corrector (Sets B/C dials: beta 0.5, eta 0.1; reuses the
#                  A/C ckpts when present, else fits once per cell);
#     2. emit    — scripts/emit_lsnpc_stream.py: per-row corrected label +
#                  the five oracle-free scores over the retrain pool
#                  (cs+tr rows; dopanim: the FULL 10,484-row pool) AND the
#                  held-out val/eval rows;
#     3. retrain — scripts/retrain_e_streams.py: downstream frozen-encoder
#                  + MLP heads, seeds 42-46, configurations:
#                    a original noisy | b blanket-corrected |
#                    c gated-corrected @coverage | d clean-oracle ceiling |
#                    e Co-teaching small-loss (pre-committed; GCE via
#                    CO_METHOD=gce) | f random-rejection-at-equal-coverage;
#                  clean-test accuracy + macro-F1 per config/seed;
#     4. analyze — scripts/analyze_set_e.py: paired-bootstrap deltas over
#                  the downstream seeds (2000 resamples).
#
#   Text primary: NoisyAG worst (med/best = severity curve); supporting:
#   CIFAR-10N worse + CIFAR-100N fine (image, frozen RN50 embedding
#   corrector). dopanim (primary, image): by default the reported FULL path
#   (pixel Swin at 224px over the full annotated pool, no clean-label
#   ceiling). DOPANIM_FULL=0 selects the (a)-only abstention path instead.
#
#   Fixed per cell (E4): gate axis text = minimality_self, image =
#   robustness_self; plausibility floor alpha=0.1 on image; operating coverage 0.1
#   with the coverage grid {0.1, 0.5, 0.9} reported unconditionally; config (e) =
#   Co-teaching (small-loss schedule R(t) warming over the first 10% of
#   steps to R_final = clamp(mis_frac+0.05, 0.2, 0.5)); GCE only via
#   CO_METHOD=gce. NEVER swap silently after seeing results.
#
# One GPU process at a time (serial). Every stage is idempotent: emissions
# and cell jsons are skipped when they already exist; the corrector ckpt is
# reused. Nothing runs until you launch it.
#
# Usage:  bash shells/set_e_run.sh [text|image|dopanim|all]
# Env overrides (examples):
#   TESTBEDS="noisyag_worst cifar10n"  SEEDS_DOWN="42 43"
#   COVERAGES="0.1 0.25 0.5"  CONFIGS="a b c d e f"  CO_METHOD=coteach
#   CORR_SEEDS="42 43 44 45 46"  FIT_IF_MISSING=1  DOPANIM_FULL=0
# dopanim cells are defined at 224px: rebuild/probe the pickles first with
#   bash shells/set_c_run.sh cc   (DOPANIM_ANY_SIZE=1 only for unquoted runs)
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
  text|image|all|dopanim) ;;
  *) echo "usage: $0 [text|image|dopanim|all]"; exit 2 ;;
esac

LOGDIR=/tmp/set_e_logs
SETEDIR="$PROJECT_ROOT/results/set_e"
CJSON="$PROJECT_ROOT/results/set_c/json"
mkdir -p "$LOGDIR" "$SETEDIR/cells" "$SETEDIR/_features"
RUN_LOG="$LOGDIR/set_e.log"
: > "$RUN_LOG"

# ── protocol constants (env-overridable) ───────────────────────────────
SEEDS_DOWN="${SEEDS_DOWN:-42 43 44 45 46}"      # downstream head seeds (paired)
CORR_SEEDS="${CORR_SEEDS:-42 43 44 45 46}"            # corrector seed(s), paired across downstream runs
MODELS="${MODELS:-mlp,linear,asl}"                    # downstream families: mlp,linear,asl
COVERAGES="${COVERAGES:-0.1 0.5 0.9}"                  # coverage grid (c/f per coverage)
CONFIGS="${CONFIGS:-a b c d e f}"
CO_METHOD="${CO_METHOD:-coteach}"         # pre-committed: coteach (gce opt-in)
FIT_IF_MISSING="${FIT_IF_MISSING:-1}"
ROB_K="${ROB_K:-32}"
EPOCHS_TEXT="${EPOCHS_TEXT:-15}"
EPOCHS_IMAGE="${EPOCHS_IMAGE:-10}"
CLEAN_SET="${CLEAN_SET:-2000}"
VAL_SIZE="${VAL_SIZE:-2500}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"; HEAD_LAYERS="${HEAD_LAYERS:-3}"
EPOCHS_HEAD="${EPOCHS_HEAD:-40}"; LR_HEAD="${LR_HEAD:-1e-3}"; BATCH_HEAD="${BATCH_HEAD:-512}"
TESTBEDS="${TESTBEDS:-}"

CKPT_TEXT="$PROJECT_ROOT/results/ckpt/lsnpc/text"
CKPT_IMAGE="$PROJECT_ROOT/results/ckpt/lsnpc/image"

total=0; ok=0; fail=0
FAIL_MSG=""
# stdout must stay clean: ensure_*_ckpt is called as $(...) and its ONLY
# stdout must be the checkpoint path. tee still appends to the log, but
# its stdout is redirected to stderr so log lines cannot contaminate a
# captured value (this silently corrupted --ckpt on every fit path).
note() { echo "$*" | tee -a "$RUN_LOG" >&2; }
die_on_fail() {
  if [ -n "$FAIL_MSG" ]; then
    note "ABORT: $FAIL_MSG"
    exit 1
  fi
}

text_testbed_sev() { echo "${1#noisyag_}"; }   # noisyag_worst -> worst
text_eval_slice() {
  case "${1#noisyag_}" in
    worst) echo 3000 ;; med) echo 2000 ;; best) echo 2500 ;;
    *) echo 2000 ;;
  esac
}

# ── corrector resolution ────────────────────────────────────────────────
text_ckpt() { # testbed corr_seed -> path
  local sev; sev=$(text_testbed_sev "$1")
  echo "$CKPT_TEXT/lsnpc_noisyag_${sev}_s$2_n0.0_symmetric_e${EPOCHS_TEXT}_b0.5_cs${CLEAN_SET}.pt"
}
cifar_ckpt() { # dataset corr_seed -> path
  local view=fine
  [ "$1" = "cifar10n" ] && view=worse
  echo "$CKPT_IMAGE/lsnpc_$1_s$2_e${EPOCHS_IMAGE}_b0.5_cs${CLEAN_SET}_${view}.pt"
}

ensure_text_ckpt() { # testbed corr_seed
  local tb="$1" cs="$2" ckpt; ckpt=$(text_ckpt "$tb" "$cs")
  [ -f "$ckpt" ] && { echo "$ckpt"; return 0; }
  if [ "$FIT_IF_MISSING" != "1" ]; then
    FAIL_MSG="missing corrector ckpt $ckpt (set FIT_IF_MISSING=1 to fit)"
    die_on_fail; return 1
  fi
  local ev; ev=$(text_eval_slice "$tb")
  note "fitting missing text corrector $tb s$cs (A dials; ckpt = $ckpt)"
  $ACCEL launch --mixed_precision=bf16 -m scripts.text_lsnpc_sst2 \
    --dataset "$tb" --noise 0.0 --noise-type symmetric \
    --epochs "$EPOCHS_TEXT" --beta 0.5 --seed "$cs" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" > "$LOGDIR/fit_text_${tb}_s${cs}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
    FAIL_MSG="text corrector fit failed for $tb s$cs (log: $LOGDIR/fit_text_${tb}_s${cs}.log)"
    die_on_fail; return 1
  fi
  echo "$ckpt"
}

ensure_cifar_ckpt() { # dataset corr_seed
  local tb="$1" cs="$2" ckpt; ckpt=$(cifar_ckpt "$tb" "$cs")
  [ -f "$ckpt" ] && { echo "$ckpt"; return 0; }
  if [ "$FIT_IF_MISSING" != "1" ]; then
    FAIL_MSG="missing corrector ckpt $ckpt (set FIT_IF_MISSING=1 to fit)"
    die_on_fail; return 1
  fi
  local view=fine; [ "$tb" = "cifar10n" ] && view=worse
  local ev=3000
  note "fitting missing frozen-RN50 corrector $tb $view s$cs"
  $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
    --dataset "$tb" --noise-type "$view" \
    --epochs "$EPOCHS_IMAGE" --beta 0.5 --seed "$cs" \
    --clean-set-size "$CLEAN_SET" --val-size "$VAL_SIZE" \
    --eval-slice "$ev" --rob-k "$ROB_K" \
    --backbone resnet50 --batch-size 256 > "$LOGDIR/fit_img_${tb}_s${cs}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
    FAIL_MSG="image corrector fit failed for $tb s$cs (log: $LOGDIR/fit_img_${tb}_s${cs}.log)"
    die_on_fail; return 1
  fi
  echo "$ckpt"
}

# ── emission ────────────────────────────────────────────────────────────
emit_cell() { # tag ckpt testbed
  local tag="$1" ckpt="$2" tb="$3"
  local cs; cs=${tag##*_cs}
  local npz="$SETEDIR/$tag/emissions_cs${cs}.npz"
  if [ -f "$npz" ]; then note "[skip-emit] $tag"; return 0; fi
  total=$((total+1)); note "=== [E EMIT] $tag ($ckpt) ==="
  $PY -m scripts.emit_lsnpc_stream --ckpt "$ckpt" \
    --outdir "$SETEDIR/$tag" --rob-k "$ROB_K" \
    > "$LOGDIR/emit_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$npz" ]; then
    FAIL_MSG="emission failed for $tag (log: $LOGDIR/emit_${tag}.log)"
    die_on_fail; return 1
  fi
  ok=$((ok+1)); note "  OK $npz"
}

# ── retrain ─────────────────────────────────────────────────────────────
retrain_cell() { # tag testbed [extra args...]
  local tag="$1" tb="$2"; shift 2
  local cs; cs=${tag##*_cs}
  local json="$SETEDIR/cells/$tag.json"
  if [ -f "$json" ]; then
    # Skip only when requested families, configurations and seed arrays are complete.
    local missing
    missing=$($PY -m scripts.check_e_cell "$json" --models "$MODELS" --seeds "$SEEDS_DOWN" --coverages "$COVERAGES" --configs "$CONFIGS")
    if [ "$missing" -eq 0 ]; then note "[skip-retrain] $tag"; return 0; fi
    note "[extend-retrain] $tag (missing families in MODELS)"
  fi
  total=$((total+1)); note "=== [E RETRAIN] $tag (coverage=$COVERAGES configs=$CONFIGS seeds=$SEEDS_DOWN) ==="
  $PY -m scripts.retrain_e_streams \
    --tag "$tag" \
    --emissions "$SETEDIR/$tag/emissions_cs${cs}.npz" \
    --testbed "$tb" --configs "$CONFIGS" --coverages "$COVERAGES" --seeds "$SEEDS_DOWN" \
    --head-hidden "$HEAD_HIDDEN" --head-layers "$HEAD_LAYERS" \
    --epochs "$EPOCHS_HEAD" --lr "$LR_HEAD" --batch "$BATCH_HEAD" \
    --models "${MODELS:-mlp}" \
    --co-method "$CO_METHOD" "$@" > "$LOGDIR/retrain_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
    FAIL_MSG="retrain failed for $tag (log: $LOGDIR/retrain_${tag}.log)"
    die_on_fail; return 1
  fi
  ok=$((ok+1)); note "  OK $json"
}

# ── text phase (NoisyAG worst primary; med/best severity) ───────────────
text_phase() {
  local tbs=(noisyag_worst noisyag_med noisyag_best)
  [ -n "$TESTBEDS" ] && tbs=($TESTBEDS)
  local tb cs ckpt tag
  for tb in "${tbs[@]}"; do
    case "$tb" in
      noisyag_worst|noisyag_med|noisyag_best) ;;
      *) continue ;;
    esac
    for cs in $CORR_SEEDS; do
      ckpt=$(ensure_text_ckpt "$tb" "$cs") || return 1
      tag="e_text_${tb}_cs${cs}"
      emit_cell "$tag" "$ckpt" "$tb"
      retrain_cell "$tag" "$tb"
    done
  done
}

# ── image phase (CIFAR-10N worse / CIFAR-100N fine, frozen RN50) ────────
image_phase() {
  local tbs=(cifar10n cifar100n)
  [ -n "$TESTBEDS" ] && tbs=($TESTBEDS)
  local tb cs ckpt tag
  for tb in "${tbs[@]}"; do
    case "$tb" in
      cifar10n|cifar100n) ;;
      *) continue ;;
    esac
    for cs in $CORR_SEEDS; do
      ckpt=$(ensure_cifar_ckpt "$tb" "$cs") || return 1
      tag="e_image_${tb}_cs${cs}"
      emit_cell "$tag" "$ckpt" "$tb"
      retrain_cell "$tag" "$tb"
    done
  done
}

# ── dopanim phase ───────────────────────────────────────────────────────
# Gateability (C-c, pre-committed): a 224px frozen-Swin cell is gateable iff
# it produces >=200 edited rows AND, at the operating point, >=10% coverage on
# the transferred gate. Scan the Set C json archive.
cc_verdict_json() { # -> path to a verdict json (or empty if no cells)
  local best=""
  local f
  for f in "$CJSON"/image_dopanim_*_sz224*.json; do
    [ -f "$f" ] || continue
    if [ -z "$best" ]; then best="$f"; continue; fi
    best="$f"   # last one scanned; the verdict function below picks cells
  done
  if [ -n "$best" ]; then echo "$best"; fi
}

dopanim_prep_size() { # prints the current dopanim train-pickle size (0 if absent)
  # Absence is a legitimate "not prepped yet" state -> 0. A present but
  # unreadable pickle must fail loudly, never be guessed as a size.
  if [ ! -f data/dopanim/train_batch ]; then echo 0; return 0; fi
  "$PY" - <<'PY'
import pickle
d = pickle.load(open("data/dopanim/train_batch", "rb"))
print(d["data"].shape[-1])
PY
}

dopanim_phase() {
  # Resolution guard: dopanim is fixed at 224px (protocol setting); every
  # dopanim E cell (abstention (a)-only included) is defined at that
  # resolution. Escape hatch: DOPANIM_ANY_SIZE=1 (unquoted probe only).
  if [ "${DOPANIM_ANY_SIZE:-0}" != "1" ]; then
    local dsz
    dsz=$(dopanim_prep_size) || {
      FAIL_MSG="dopanim train pickle present but unreadable (see traceback above)"
      die_on_fail; return 1
    }
    if [ "$dsz" != "224" ]; then
      note "dopanim pickles are ${dsz}px; dopanim E cells are defined at 224px."
      note "  rebuild + probe with:  bash shells/set_c_run.sh cc"
      note "  (then re-run this; or DOPANIM_ANY_SIZE=1 for an unquoted probe)"
      return 0
    fi
  fi
  if [ "${DOPANIM_FULL:-1}" != "1" ]; then
    note "=== dopanim ABSTENTION path (DOPANIM_FULL=0) ==="
    note "  (no gateable 224px C-c cell pre-committed; run C-c first, then"
    note "   set DOPANIM_FULL=1 when a cell passes the gateability gate)"
    local cs tag verdict_json cell_json
    local cc=""; cc=$(cc_verdict_json)
    # Seed list follows CORR_SEEDS (default is five seeds 42 through 46).
    for cs in $CORR_SEEDS; do
      tag="e_image_dopanim_cs${cs}"
      verdict_json=""
      if [ -n "$cc" ]; then
        verdict_json="$SETEDIR/$tag/gate_verdict.json"
        mkdir -p "$SETEDIR/$tag"
        $PY - "$cc" "$verdict_json" <<'PY'
import json, sys
src = sys.argv[1]; dst = sys.argv[2]
r = json.load(open(src))
t = r.get("transitions") or {}
wr = int(t.get("WR", 0)); rw = int(t.get("RW", 0))
edits = wr + rw
n_eval = int(r.get("n_eval", 0))
cov = (edits / n_eval) if n_eval else 0.0
gateable = edits >= 200 and cov >= 0.10
verdict = {"cell": src, "WR": wr, "RW": rw, "edits": edits,
           "edit_rate_eval": cov, "net": wr - rw,
           "gateable": gateable,
           "verdict": ("GATEABLE" if gateable and (wr - rw) >= 0 else
                       "gateable but net-negative" if gateable else
                       "ABSTAINING / not gateable")}
json.dump(verdict, open(dst, "w"), indent=2)
print("  gateability accounting:", json.dumps(verdict))
PY
        local grc=$?
        if [ "$grc" -ne 0 ]; then
          FAIL_MSG="gateability accounting failed on $cc (exit $grc); fix the C-c json"
          die_on_fail; return 1
        fi
      fi
      local json="$SETEDIR/cells/$tag.json"
      local need=1
      if [ -f "$json" ]; then
        need=$($PY -m scripts.check_e_cell "$json" --models "$MODELS" --seeds "$SEEDS_DOWN" --coverages "$COVERAGES" --configs "a")
      fi
      if [ "$need" -eq 1 ]; then
        total=$((total+1)); note "=== [E DOPANIM (a)-ONLY] $tag ==="
        $PY -m scripts.retrain_e_streams --tag "$tag" --testbed dopanim \
          --seeds "$SEEDS_DOWN" --a-only --models "${MODELS:-mlp}" \
          --gate-file "$verdict_json" \
          --head-hidden "$HEAD_HIDDEN" --head-layers "$HEAD_LAYERS" \
          --epochs "$EPOCHS_HEAD" --lr "$LR_HEAD" --batch "$BATCH_HEAD" \
          > "$LOGDIR/retrain_${tag}.log" 2>&1
        local rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
          FAIL_MSG="dopanim (a)-only retrain failed (log: $LOGDIR/retrain_${tag}.log)"
          die_on_fail; return 1
        fi
        ok=$((ok+1)); note "  OK $json"
      else
        note "[skip-retrain] $tag"
      fi
    done
    return 0
  fi
  note "=== dopanim FULL path (gateable C-c cell available) ==="
  note "  corrector fit: frozen Swin @224 over the FULL 10,484-row pool (--pool-full), split seed 42 (index-fixed), eval 3000 / val 2500; five corrector seeds by default. Retrain pool = full pool."
  local cs ckpt tag
  # Seed list follows CORR_SEEDS (default is five seeds 42 through 46);
  # in this branch a missing corrector is FITTED (FIT_IF_MISSING=1).
  for cs in $CORR_SEEDS; do
    tag="e_image_dopanim_cs${cs}"
    ckpt="$CKPT_IMAGE/lsnpc_dopanim_s${cs}_e${EPOCHS_IMAGE}_b0.5_cs${CLEAN_SET}_eta0.1_swin_sz224_symmetric.pt"
    if [ ! -f "$ckpt" ]; then
      if [ "$FIT_IF_MISSING" != "1" ]; then
        FAIL_MSG="missing dopanim corrector $ckpt"; die_on_fail; return 1
      fi
      note "fitting dopanim corrector s$cs (pixel Swin 224, FULL pool, --split-seed 42)"
      $ACCEL launch --mixed_precision=bf16 -m scripts.image_lsnpc \
        --dataset dopanim --noise-type symmetric \
        --epochs "$EPOCHS_IMAGE" --beta 0.5 --eta 0.1 --seed "$cs" \
        --split-seed 42 --allow-large-eval --eval-slice 3000 \
        --val-size "$VAL_SIZE" --clean-set-size "$CLEAN_SET" --rob-k "$ROB_K" \
        --pixel --pool-full --backbone swin --batch-size 64 \
        > "$LOGDIR/fit_dopanim_s${cs}.log" 2>&1
      if [ $? -ne 0 ] || [ ! -f "$ckpt" ]; then
        FAIL_MSG="dopanim corrector fit failed s$cs (log: $LOGDIR/fit_dopanim_s${cs}.log)"
        die_on_fail; return 1
      fi
    fi
    emit_cell "$tag" "$ckpt" dopanim
    retrain_cell "$tag" dopanim --no-clean-ceiling
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
validate_seed_list "$CORR_SEEDS" || exit 2
validate_seed_list "$SEEDS_DOWN" || exit 2

if [ "${DRY_RUN:-0}" = 1 ]; then
  for tb in ${TESTBEDS:-noisyag_worst noisyag_med noisyag_best cifar10n cifar100n dopanim}; do
    case "$tb" in noisyag_*) part=text;; cifar10n|cifar100n) part=image;; dopanim) part=dopanim;; *) echo "unknown testbed $tb" >&2; exit 2;; esac
    [ "$MODE" = all ] || [ "$MODE" = "$part" ] || continue
    for cs in $CORR_SEEDS; do
      for ds in $SEEDS_DOWN; do
        printf 'PAIR %s corrector=%s downstream=%s models=%s coverage=%s\n' "$tb" "$cs" "$ds" "$MODELS" "$COVERAGES"
      done
    done
  done
  exit 0
fi

# ── dispatch ─────────────────────────────────────────────────────────────
case "$MODE" in
  text) text_phase ;;
  image) image_phase ;;
  dopanim) dopanim_phase ;;
  all) text_phase; image_phase; dopanim_phase ;;
esac

note "=== Set E ($MODE): ok=$ok new=$total fail=$fail ==="
if [ "$fail" -ne 0 ] || [ -n "$FAIL_MSG" ]; then
  note "ERROR: $FAIL_MSG — logs under $LOGDIR"
  exit 1
fi
note "running the E analyzer over results/set_e/cells ..."
$PY -m scripts.analyze_set_e --cells "$SETEDIR/cells" \
  --out "$SETEDIR/analysis" 2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || exit "$rc"
note "analysis: $SETEDIR/analysis/summary.json"
