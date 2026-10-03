# When Do Noisy-Label Corrections Help? A Counterfactual Audit

Anonymous Authors

This repository contains the anonymized research code accompanying the paper
**When Do Noisy-Label Corrections Help? A Counterfactual Audit**. It provides the
implementation and experiment scripts associated with the paper for anonymous
peer review.

## Overview

The paper studies when proposed corrections to noisy training labels help
subsequent learning. It applies latent-shift noisy-prediction correction (LSNPC)
to training labels and audits the proposed corrections using scores based on
four counterfactual properties: **validity, proximity, noise-robustness, and
plausibility**.

The pipeline generates candidate label corrections, scores and ranks them, and
uses a rank threshold to retain or revert corrections before downstream
retraining. A latent path that changes the decoder's label is treated as a
counterfactual candidate. **Label-free** refers to scoring and selection;
corrector training and clean-label diagnostics are separate.

The A–E experiment entrypoints cover score auditing, selective correction,
mechanism analysis, clean-label budgets, and downstream retraining. Refer to the
paper for the methodology, experimental protocols, and reported results.

## Repository structure

| Path | Contents |
|---|---|
| `data_process/` | Dataset loading, label processing, and noise handling |
| `experiments/` | Experiment configuration, scoring, and pipeline logic |
| `models/` | Corrector, encoder, and model definitions |
| `trainers/` | Training loops |
| `scripts/` | Data preparation, training, stream emission, and analysis tools |
| `shells/` | A–E experiment entrypoints |
| `utils/` | Batching, resource paths, serialization, and run metadata |
| `tests/` | Model, protocol, and portability tests |
| `requirements.txt` | Python dependencies |

## Installation

The experiment launchers use **Linux, Bash, and an NVIDIA CUDA GPU**. Set B is
an offline analysis that can run on a CPU. Use a dedicated Python environment
with compatible PyTorch, torchvision, and CUDA versions.

From the repository root, in the activated environment, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Installation should finish without dependency errors. If it fails, first check
the Python version and the PyTorch/torchvision/CUDA combination.

The launchers use `python` and `accelerate` from `PATH`. Set `PY` and `ACCEL` to
select different executables. Run Python tools as modules from the repository
root, using `python -m scripts.<name>`.

## Data and pretrained models

Place datasets under `data/` and pretrained models in the paths below. Obtain
them from their respective providers and retain the applicable terms, licenses,
and attribution. Preserve row alignment between inputs and noisy/clean labels.

| Dataset | Expected input | Preparation |
|---|---|---|
| SST-2 | `data/sst2/train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` | `python -m scripts.text_sst2_prep` writes `embeddings.pkl`; A uses the labeled training pool |
| AG News | `data/ag_news/train_emb.pkl`, containing feature matrix `x` and labels `y` | Supply an MPNet embedding cache with aligned rows and labels |
| NoisyAG-News | `data/noisyag_news/noisyag_best.jsonl`, `noisyag_med.jsonl`, and `noisyag_worst.jsonl`; records contain `text`, `ground_truth`, and `noisy_label` | `python -m scripts.noisyag_prep` writes the embedding cache |
| Medical abstracts | `data/medical_abstracts/train.jsonl` and `test.jsonl`; records contain `text` and labels 1–5 | `python -m scripts.medical_prep` writes the embedding cache |
| CIFAR-10N | Original CIFAR-10 archive and `data/CIFAR-10N_human.pt` | `python -m scripts.rebuild_cifar_original_order --help` describes preparation in the original row order |
| CIFAR-100N | CIFAR-100 parquet and `data/CIFAR-100_human.pt` | `python -m scripts.rebuild_cifar100_from_parquet --help` describes preparation and label alignment |
| EuroSAT | `data/eurosat/{train,val,test}.parquet` | See `data_process/eurosat.py` for the image-column format |
| dopanim | `data/dopanim/annotation_data.json` and JPEGs under `extracted/train/<class>/` and `extracted/test/<class>/` | `python -m scripts.dopanim_prep --size 224` writes the image pickles |

Run preparation commands from the repository root. Preparation produces the
embedding or image caches consumed by the experiment scripts. If a cache cannot
be created or loaded, check the filenames, schemas, and label alignment first.
The CIFAR human-label releases are documented in
[UCSC-REAL/cifar-10-100n](https://github.com/UCSC-REAL/cifar-10-100n).

Pretrained model locations:

- `models/all-mpnet-base-v2/`: the
  `sentence-transformers/all-mpnet-base-v2` snapshot, including tokenizer,
  configuration, and SentenceTransformer files.
- `local_models/swin-base-patch4-window7-224/`,
  `local_models/swin-tiny-patch4-window7-224/`, and
  `local_models/vit-base-patch16-224/`: the selected Swin/ViT snapshots.
  `IMAGE_MODEL_DIR` can override the parent directory; Swin/ViT loading uses
  local files.
- ResNet features use torchvision's pretrained weight cache.

Keep the model identities and tokenization consistent with the selected
experiment configuration. Dataset caches, model weights, and generated results
are excluded by `.gitignore`.

## A–E experiments

| Set | Purpose | Entrypoint |
|---|---|---|
| A | Score inventory and recovery-regime analysis | `shells/set_a_run.sh` |
| B | Selective-correction and matched random-rejection analysis | `shells/set_b_run.sh` |
| C | Beta/eta, encoder, clean-budget, and resolution studies | `shells/set_c_run.sh` |
| D | Nested clean-label budget studies | `shells/set_d_run.sh` |
| E | Corrected label-stream emission and downstream retraining | `shells/set_e_run.sh` |

The launchers default to training seeds **42, 43, 44, 45, and 46**. Set E uses
separate corrector and downstream seed lists, controlled by `CORR_SEEDS` and
`SEEDS_DOWN`. Data-split and bootstrap seeds serve different roles.

Run the commands below from the repository root on the configured Linux GPU
server; B can run on a CPU. Run one campaign at a time because temporary run
directories are shared by the scripts.

### Set A: score auditing

```bash
bash shells/set_a_run.sh all
```

A produces score summaries and per-query outputs under `results/set_a/`, and
corrector checkpoint bundles under `results/ckpt/lsnpc/`. Modes `text` and
`image` restrict the modality. For a failed cell, inspect `/tmp/set_a_logs/`
and check its dataset and model files.

### Set B: selective correction

```bash
bash shells/set_b_run.sh all
```

B consumes A's JSON/NPZ outputs and writes selection curves and summaries under
`results/set_b/`. It checks seed coverage before analysis. On failure, first
check for missing JSON/NPZ pairs or missing seeds in A. `SET_A_JSON` and
`SET_A_PER_QUERY` select alternative input collections.

### Set C: mechanism analysis

```bash
bash shells/set_c_run.sh all
```

C writes its outputs under `results/set_c/`. Modes `ca_text`, `ca_image`, `cb`,
`cc`, and `cc_rn50` select individual studies. The dopanim probes use 224 px
inputs; keep the raw files available for pickle preparation. Inspect
`/tmp/set_c_logs/` and missing input files if a cell fails.

### Set D: clean-label budgets

```bash
bash shells/set_d_run.sh all
```

D produces outputs under `results/set_d/` for nested budgets of 100, 400, and
2000 clean labels. Its defaults are 15 text epochs and 5 image epochs. Inspect
`/tmp/set_d_logs/` and the selected input data on failure.

After collecting D outputs, aggregate them from the repository root:

```bash
python -m scripts.analyze_set_d_budget --json-dir results/set_d/json --out results/set_d
```

The analyzer reports the budgets found and writes summary files. If the expected
budgets are absent, check the input directory and epoch settings.

### Set E: downstream retraining

```bash
bash shells/set_e_run.sh all
```

E fits or loads correctors, emits corrected label streams, and fits downstream
models. Outputs are stored under `results/set_e/`. It includes MLP, linear, and
ASL downstream models with coverages 0.1, 0.5, and 0.9. Modes `text`, `image`,
and `dopanim` restrict the testbeds. Inspect `/tmp/set_e_logs/`, data files, and
checkpoint bundles if a cell fails.

The configuration letters denote (a) noisy labels, (b) unselective correction,
(c) thresholded correction, (d) the clean-label reference where available,
(e) Co-teaching, and (f) matched random rejection. Co-teaching and random
rejection are MLP controls. The dopanim setting uses a pixel-mode Swin corrector
at 224 px over the full annotated pool and has no clean-label reference.

## Tests

From the repository root, in the configured Python environment:

```bash
python -m pytest -q tests
```

The suite checks model behavior, score calculations, split policies, batching,
seed integrity, and resource-path portability. A successful run has no failed
tests; investigate the reported assertion and dependencies if a test fails.

## Notes on optional analyses

- In Set B, diagnostic rules using a non-trivial plausibility floor have a known
  eligibility-mask ordering issue in `scripts/analyze_set_b_gates.py`. The paper
  excludes the affected diagnostic conjunction; rules without a floor are
  unaffected.
- Optional paired score-reference reporting with `EMIT_FRAMES=1` supports text
  and frozen-ResNet-50 cells. Use `IMAGE_BACKBONES=resnet50` for this mode, as the
  pixel path does not pass its VAE latents to that optional report.
