# Label-Correction Experiments

Anonymous Authors

Research code for **Latent-Shift Noisy Prediction Correction (LSNPC)** and the
A–E experiments on label noise, correction scores, selective correction,
clean-label budgets, and downstream retraining. This distribution exposes five
experiment bash entrypoints. It does not reproduce the separate pool-expansion,
extended-candidate, or ten-seed robustness studies.

**Status:** local preparation and tests are complete. Training data, downloaded
backbones, checkpoints, and published result archives are not included. No full
training campaign has been run for this distribution. Five-seed configuration
coverage is not a claim that five-seed results have been generated. See
[reproduction scope and known differences](#reproduction-scope-and-known-differences) for unresolved manuscript/configuration
conflicts and the precise scope of verification.

## Layout

| Directory | Contents |
|---|---|
| `shells/` | Five experiment entrypoints: A, B, C, D, E |
| `scripts/` | Training, emission, data preparation and analysis tools; ancillary Python helpers are retained |
| `data_process/` | Dataset loading and noise handling |
| `models/`, `trainers/` | Model definitions and training loops |
| `experiments/` | Shared configuration, scoring and pipeline logic |
| `utils/` | Batching, project-relative resource paths and run provenance |
| `tests/` | Unit tests, relocation checks and seed-cache checks |

## Environment

The full experiment bash scripts target **Linux with an NVIDIA CUDA GPU** and
Bash. Use a dedicated Python environment. Select a PyTorch/torchvision build
compatible with that GPU and CUDA installation before installing the remaining
dependencies. `requirements.txt` is a dependency list, not a locked training
environment; the original training environment has not been independently
reconstructed here.

In your activated environment, from this repository root:

```bash
python -m pip install -r requirements.txt
```

This installs the listed dependencies. A successful installation finishes without
resolver errors. If it fails, check the selected Python version and the
PyTorch/torchvision/CUDA combination before attempting training.

See [verification status](#verification-status) for the recorded test environment
and checks. Unit tests do not establish CUDA training compatibility or numerical
reproduction.

The launchers find `python` and `accelerate` on `PATH`. To select another active
environment, set `PY` and `ACCEL` to its executables. Resource paths are resolved
from this directory, without a fallback to another research checkout. Run
Python tools as modules (`python -m scripts.<name>`) from the root.

## Data and backbones

Place inputs in `data/` and models in the locations below. These directories are
ignored by Git. Obtain datasets and model files from their respective providers;
retain their terms, licenses and attribution. Do not substitute differently
ordered rows for the human-label files.

| Input | Expected files or structure | Preparation |
|---|---|---|
| SST-2 | `data/sst2/train-00000-of-00001.parquet`, `test-00000-of-00001.parquet` | `python -m scripts.text_sst2_prep` writes `embeddings.pkl`. The A pipeline uses its labeled **training** pool; standard held-out test labels are not a clean evaluation oracle. |
| AG News | `data/ag_news/train_emb.pkl` with `x` (float32 feature matrix), `y` (integer labels) | Supply an MPNet embedding cache in the same row order as its labels. An AG News raw-data embedding builder is not included. |
| NoisyAG-News | `data/noisyag_news/noisyag_best.jsonl`, `noisyag_med.jsonl`, `noisyag_worst.jsonl`; each record has `text`, `ground_truth`, `noisy_label` | `python -m scripts.noisyag_prep` checks shared row order and writes `embeddings.pkl`. |
| Medical abstracts | `data/medical_abstracts/train.jsonl`, `test.jsonl`; records have `text`, labels 1–5 | `python -m scripts.medical_prep` writes the embedding cache. |
| CIFAR-10N | Original CIFAR-10 archive and `data/CIFAR-10N_human.pt` | `python -m scripts.rebuild_cifar_original_order --help`; preserve original ordering when creating the local pickles. |
| CIFAR-100N | `data/CIFAR-100_human.pt` and CIFAR-100 parquet | `python -m scripts.rebuild_cifar100_from_parquet --help`; verifies label alignment and produces `cifar100_train.pkl` / `cifar100_test.pkl`. |
| EuroSAT | `data/eurosat/{train,val,test}.parquet`, from the `blanchon/EuroSAT_RGB` dataset representation | See `data_process/eurosat.py` for the image-column format. |
| dopanim | `data/dopanim/annotation_data.json` plus `extracted/train/<class>/` and `extracted/test/<class>/` JPEG directories | `python -m scripts.dopanim_prep --size 224` writes the train/test pickles used by C-c and E. |

The human-label CIFAR releases are documented in
[UCSC-REAL/cifar-10-100n](https://github.com/UCSC-REAL/cifar-10-100n).
Input parsers are the authoritative file-format definitions; prepared data is
not bundled. Preparation succeeds when the stated caches exist with aligned
rows and labels; on failure, first check the expected filenames and schemas.

Model locations:

- `models/all-mpnet-base-v2/`: the `sentence-transformers/all-mpnet-base-v2`
  snapshot, including tokenizer/configuration and the SentenceTransformer files
  used by preparation scripts.
- `local_models/swin-base-patch4-window7-224/`,
  `local_models/swin-tiny-patch4-window7-224/`, and
  `local_models/vit-base-patch16-224/`: the corresponding pretrained model
  snapshots when those backbones are selected. `IMAGE_MODEL_DIR` can override
  the parent directory. Swin/ViT loading is local-only.
- ResNet features use torchvision's pretrained weight cache. Ensure it is
  populated or that the first training invocation can download its weights.

Do not change model identities or tokenization simply to satisfy a missing-file
error: doing so changes the scientific configuration.

## A–E settings

All retained training grids default to **42, 43, 44, 45, 46**. The launchers
reject fewer than five distinct training seeds. More seeds may be supplied as
an explicitly separate extension. Data-split and bootstrap random seeds are
not mechanically replaced by training-seed loops.

| Set | Purpose and default scope | Main settings |
|---|---|---|
| A | Score/regime inventory: 17 text settings plus 7 image settings with each of frozen RN50 and Swin | 31 settings × 5 seeds = 155 training cells; beta 0.5; clean set 2000; validation 2500; robustness K=32 |
| B | Offline selective-correction and random-rejection analysis of A | Requires complete five-seed JSON/NPZ input for every setting; does not fit five new models |
| C | Beta/eta sweep, frozen-encoder budget comparison, dopanim resolution probe and its frozen-ResNet-50 companion | Retains the source beta/eta sweep because manuscript descriptions conflict; see [known differences](#reproduction-scope-and-known-differences) before interpreting it as a paper reproduction |
| D | Nested clean-label budgets for NoisyAG med, four Medical settings, CIFAR-10N worse and CIFAR-100N fine (both frozen RN50) | Budgets 100/400/2000; 7 settings × 3 budgets × 5 seeds = 105 cells; beta 0.5, eta 0.1; 15 text and 5 image epochs |
| E | Corrector fit/load → corrected stream emission → downstream fitting | Five corrector seeds × five downstream seeds; MLP/linear/ASL; coverages 0.1/0.5/0.9; configurations a–f |

Text training is 15 epochs; the main image training default is 10 epochs.
Two studies differ: the D image ladder uses 5 epochs, and C-c sweeps image
epochs 10 and 15 (its frozen-ResNet-50 companion uses 5). Text batch size is 512; frozen
RN50 uses 256 and pixel Swin uses 64. Robustness perturbation count K=32 and
importance-sampled latent count are different parameters. See the model and
CLI configuration for the latter, not the shell's `ROB_K`.

E's configuration letters mean: (a) noisy labels, (b) all corrections,
(c) thresholded corrections, (d) clean-label ceiling when available,
(e) Co-teaching, (f) matched random rejection. Co-teaching and the random control
are computed for the primary MLP in this implementation. Do not remove (f)
because the outer experiment suite is named A–E.

## Preview before training

From the repository root on a Mac or Linux machine with Python and Bash:

```bash
DRY_RUN=1 bash shells/set_a_run.sh all
```

This prints `CELL` entries from A's actual grid loops without loading data or
training. Change the letter to preview C/D or E (`PAIR` entries); B reports its
offline dependency. A preview may create log files and empty output directories.
Success means zero exit status and the expected cells/seeds. If not, inspect the
error and environment overrides; previews do not certify the data or checkpoints.

## Run experiments

Run each command **in the repository root on the configured Linux GPU server**.
Only B is a CPU-only analysis. Launch one campaign at a time because temporary
run directories are shared by the inherited training scripts.

```bash
bash shells/set_a_run.sh all
```

A writes `results/set_a/json/` and `per_query/`, plus checkpoint bundles under
`results/ckpt/lsnpc/`. On failure inspect the per-cell log in `/tmp/set_a_logs/`
and check the selected data/weight files. `text` and `image` restrict modality.
`EMIT_ONLY=1` re-emits from existing compatible bundles, without a new fit.
Each cell also emits softmax entropy as a robustness control.

```bash
EMIT_ONLY=1 EMIT_FRAMES=1 IMAGE_BACKBONES=resnet50 bash shells/set_a_run.sh all
```

This optional pass records the label-free variants of proximity and
noise-robustness next to the clean-label values, under `results/set_a/frames/`
and `results/set_a/frame_scores/`. It needs A's checkpoint bundles. Restrict it
to text and frozen-ResNet-50 cells; see the known issues in the
[reproduction scope and known differences](#reproduction-scope-and-known-differences).

```bash
bash shells/set_b_run.sh all
```

B checks A's seed coverage, then writes `results/set_b/keep_curves.csv` and its
summary. On failure, first fix missing JSON/NPZ pairs or missing seeds in A.
Use `SET_A_JSON` and `SET_A_PER_QUERY` to analyze another complete collection.

```bash
bash shells/set_c_run.sh all
```

C writes `results/set_c/`. Modes `ca_text`, `ca_image`, `cb`, `cc`, `cc_rn50`
select its substudies; `cc_rn50` is the unsupervised frozen-ResNet-50 run on
the same 224px dopanim stream. Check [reproduction scope and known differences](#reproduction-scope-and-known-differences) for its unresolved
manuscript differences. C-c may rebuild the dopanim image pickles at 224px;
keep your raw input files available. First check `/tmp/set_c_logs/` on failure.

```bash
bash shells/set_d_run.sh all
```

D writes `results/set_d/` for the nested budgets. Successful execution prints
100, 400 and 2000 as its budgets. It can reuse compatible A/C cells; keep
extensions with different scientific settings in separate result directories.
Inspect `/tmp/set_d_logs/` and missing input data on failure.

```bash
python -m scripts.analyze_set_d_budget --json-dir results/set_d/json --out results/set_d
```

This aggregates D outputs at 15 text and 5 image epochs by default
(`--epochs-text`, `--epochs-image`); it prints the training budgets it found.
Verify the budget/seed inventory before interpreting its summaries. Missing inputs are a failed reproduction, not zero-valued results.

```bash
bash shells/set_e_run.sh all
```

E writes emissions and downstream cells under `results/set_e/`, then performs
per-cell analysis. Cached cells are checked for the requested seed lists,
coverage levels and model families. Check `/tmp/set_e_logs/` for failures.
`CORR_SEEDS` and `SEEDS_DOWN` separately control the two five-seed lists.
dopanim runs the reported setting by default: a pixel-mode Swin corrector at
224px over the full annotated pool. Prepare the 224px pickles first (C-c does
this). The clean-data ceiling is undefined for that dataset and is omitted.
`DOPANIM_FULL=0` selects the inherited (a)-only abstention path instead. A `PAIR`
preview describes seed scheduling, not successful execution of every E arm.

## Verification status

From the root in the configured Python environment:

```bash
python -m pytest -q tests
```

- Original tracked code was fully exported and checked file by file before any
  cleanup.
- Five bash entrypoints pass syntax checks. Python files pass syntax parsing.
- Actual A/C/D grid previews cover seeds 42–46 in every retained cell family.
  A: 155 cells in 31 families. C: 485 scheduled calls (475 unique tags) in 95
  families. D: 105 unique cells in 21 families, image ladder at 5 epochs.
- E preview enumerates 150 corrector/downstream seed pairs over six testbeds;
  each testbed has a 5 × 5 seed schedule. B schedules no training.
- Unit suite: 130 passed, one inherited warning about converting a tensor with
  gradients to a scalar in a test assertion.
- Additional checks cover path relocation, exclusion of a parent repository's
  Git identity, missing/duplicate seeds, missing coverage, and incomplete B inputs.
- The model-structure test fixture disables pretrained weight downloads; it
  retains the real torchvision model architecture.
- No GPU campaign, dataset download, checkpoint numerical comparison, or
  reproduction of published table values has been performed. The C-c ResNet-50
  mode, the A frame mode and the E dopanim default were checked by preview only.
- Requirements specify ranges; this is not a locked reproduction environment.

Recorded test environment: Python 3.11.14, PyTorch 2.12.1, transformers 5.12.1,
accelerate 1.14.0, NumPy 2.4.6, pytest 9.1.1 on macOS. The training launchers
remain intended for Linux/CUDA.

## Reproduction scope and known differences

This distribution keeps the selected source implementation and makes its paths,
launchers and seed coverage portable. The manuscript, its table captions, and
archived summaries contain disagreements. None of the following discrepancies
has been resolved by inventing results or silently changing the model objective.

| Issue | Evidence and distribution behavior |
|---|---|
| Seed counts | The manuscript uses five corrector seeds for the audit, threshold, calibration and clean-budget studies, and five corrector × five downstream seeds for end-to-end retraining. The five launchers default to seeds 42–46; this is not a claim that archived tables were generated by these exact launch commands. The ten-seed robustness check is outside this release. |
| Archived seed exceptions | The score-inventory caption states that its dopanim row (the 224 px probe) has three archived seeds, and the reference-frame table uses an archived cohort of six seeds per text/CIFAR setting and three for dopanim. The launchers schedule five seeds for every retained cell; they do not recreate those archived cohorts. |
| Image epochs | The general setup specifies 15 text and 10 image epochs; A, C-a, C-b and E use these defaults. The clean-budget ladder (D) is analyzed at 5 image epochs and 15 text epochs (the source analyzer's defaults), so D's image default is 5. C-c sweeps 10 and 15 epochs; the C-c ResNet-50 companion uses 5. |
| C-a clean budget and eta | The C-a table caption describes eta 0 and clean budget 0; the C-a prose describes the main configuration's clean budget with eta 0.1. The source uses clean size 2000 and eta 0.1/0.5. These source defaults are retained and disclosed, rather than being relabeled an exact table reproduction. |
| C campaign size | The manuscript reports 415 completed runs over nine text and five image settings. The source grid with five seeds, frozen RN50/Swin C-b, the full C-c sweep and the C-c ResNet-50 companion schedules 485 calls, 475 unique tags due to compatible C-a/C-b overlap. This is not a claimed reproduction of the 415-run campaign. |
| C-c | The 224 px probe is the pixel-mode frozen Swin corrector (beta 0.1/0.5, epochs 10/15, clean set 2000, evaluation slice 1000). The frozen-ResNet-50 companion on the same stream is unsupervised (eta 0, clean set 0), 5 epochs, beta 0.1, evaluation slice 3000 with split seed 42; it runs as `set_c_run.sh cc_rn50`. |
| E clean labels | The manuscript states that the retained end-to-end result files do not record the corrector's trusted-set size and that the campaign run script defaults to 2000 clean examples. The release keeps that launcher default (2000). The label-free claim concerns scoring and selection only. |
| Score references | A reports proximity and noise-robustness against the evaluation-only clean label; E ranks with their label-free (self-referenced) variants. E emissions store both variants on the same rows. `EMIT_FRAMES=1` on A additionally records both variants for text and frozen-ResNet-50 cells (see known issues). Entropy is emitted as a robustness control beside the four scored properties. |
| Statistical units | The retained E analyzer reports per-cell bootstrap deltas over downstream seeds. Paper-wide mean/standard error over independent corrector seeds requires a second aggregation step; the retained analyzer is not advertised as producing the full manuscript table. |

### Known implementation issues (not changed)

- `scripts/analyze_set_b_gates.py` applies the plausibility-floor eligibility
  mask in pre-sorting order (`order[elig[cand_pos]]`, where
  `order[elig[cand_pos][order]]` is intended). Only gate rules with a
  non-trivial plausibility floor are affected; rules without a floor are not.
  The manuscript does not report the affected diagnostic conjunction.
- In `scripts/image_lsnpc.py`, the pixel path does not pass its VAE latents to
  the optional two-frame report (`EMIT_FRAMES=1`). Use frame mode with text and
  frozen-ResNet-50 cells only (`IMAGE_BACKBONES=resnet50`). The default
  campaign path and the E emissions are unaffected.

The nested split, clean-label usage, loss, backbone implementation, and protocol
score calculations are preserved. Additional historical Python helpers remain
for dependency compatibility, but other launchers are excluded.
This distribution documents A–E only, not full-paper numerical reproduction.

Corrections made during preparation: the budget-list calculation now reliably
produces positive integer budgets; E cache checks validate actual seed arrays and
requested arms/coverage; B checks complete A inputs before analysis. These are
launcher/integrity changes, not new research results.

Do not publish locally generated logs, checkpoints or configuration dumps
without checking their metadata for personal paths. This directory contains no
source Git history or original author contact information; no license grant is
implied beyond the applicable upstream rights.
