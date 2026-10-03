"""LSNPC label correction on SST-2 text using a frozen sentence encoder.

Uses the codebase's proper classes: ``TextEncoder`` (models/text_encoder.py)
serves both the data-prep embedder and the trainer's "VAE" latent role (its
``encode`` returns the embedding as the latent), and ``TextDataset``
(data_process/text.py) holds the embeddings + labels.

Embedding-space correction: the frozen encoder's 768-d sentence embedding is the
latent ``z``; LSNPC recenters it toward the clean class; the MLP data-decoder
reconstructs the embedding (recon loss). Synthetic label noise is injected to make
(clean, noisy) pairs; evaluation is on a held-out train slice (data-mode),
mirroring cifar10n.

Usage: python scripts/text_lsnpc_sst2.py [--noise 0.3 --epochs 30 --beta 0.5]
"""
from __future__ import annotations

from utils.paths import project_path

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,  # override any library-set root config so [LSNPC] lines show
)

from accelerate import Accelerator

from experiments.arguments import ExperimentConfig
from experiments.lsnpc_stage1 import _resolve_lsnpc_eta, resolve_clean_rows
from experiments.protocol import (
    SCORE_KEYS,
    compute_protocol_scores,
    inject_noise,
    inject_noise_idn,
    multiclass_f1,
    multiclass_precision,
    multiclass_recall,
)
from experiments.deployed_frame import emit_frame_report
from models.text_encoder import build_text_encoder
from scripts.lsnpc_ckpt import bundled_build_kwargs, load_bundle, save_bundle
from trainers.lsnpc import LSNPCTrainer

DATA_ROOT = Path(project_path('data'))


def load_embeddings(dataset: str):
    """Load cached frozen embeddings for the text datasets.

    noisyag_* return a dict with ``X``, ``y_clean`` (ground_truth oracle) and
    ``y_noisy`` (the variant's real crowd noisy labels); sst2/ag_news/
    medical_abstracts return ``(X, yc)`` with synthetic noise injected by
    the caller.
    """
    if dataset == "sst2":
        c = pickle.load(open(DATA_ROOT / "sst2" / "embeddings.pkl", "rb"))
        return c["train_x"], c["train_y"]
    if dataset == "ag_news":
        tr = pickle.load(open(DATA_ROOT / "ag_news" / "train_emb.pkl", "rb"))
        return tr["x"], tr["y"]
    if dataset == "medical_abstracts":
        c = pickle.load(open(DATA_ROOT / "medical_abstracts" / "embeddings.pkl",
                             "rb"))
        return c["x"], c["y"]
    if dataset in ("noisyag_best", "noisyag_med", "noisyag_worst"):
        c = pickle.load(open(DATA_ROOT / "noisyag_news" / "embeddings.pkl", "rb"))
        variant = dataset.split("_", 1)[1]
        return {"X": c["x"], "y_clean": c["ground_truth"],
                "y_noisy": c[f"noisy_{variant}"]}
    raise ValueError(dataset)


def build_config(seed, epochs, beta, noise, batch, clean_set_size, eval_slice, latent_dim, dataset, n_classes, eta=None, correction_cond: str = "none"):
    sys.argv = ["text"] + [
        "--dataset", dataset, "--noise", str(noise), "--noise-type", "symmetric",
        "--seed", str(seed), "--conditioning-mode", "posterior",
        "--use-semi", "--lsnpc-beta", str(beta), "--lsnpc-lr", "5e-4",
        "--clean-set-size", str(clean_set_size), "--lsnpc-epochs", str(epochs),
        "--n-test", "-1", "--yhat-source", "data",
        "--eval-train-slice", str(eval_slice),
        "--encoder-backbone", "conv", "--batch-size", str(batch),
        "--output-dir", f"/tmp/text_{dataset}_s{seed}_n{noise}_e{epochs}_b{beta}_cs{clean_set_size}",
        "--latent-dim", str(latent_dim), "--hidden-dim", "256",
        "--n-blocks", "4", "--iw-samples", "5",
    ]
    if eta is not None:
        sys.argv += ["--lsnpc-eta", str(eta)]
    if correction_cond != "none":
        sys.argv += ["--correction-cond", str(correction_cond)]
    config = ExperimentConfig.from_args()
    config.validate()
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    config.lsnpc_image_data = False
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sst2",
                    choices=["sst2", "ag_news", "noisyag_best", "noisyag_med",
                             "noisyag_worst", "medical_abstracts"])
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--noise-type", default="symmetric", choices=["symmetric", "idn"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="RNG seed for the eval/val/cs split carves. Default: "
                         "--seed (Set E keeps splits reproducible across "
                         "corrector/downstream seeds).")
    ap.add_argument("--eval-slice", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--clean-set-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512,
                    help="LSNPC/VAE batch on cached embeddings (default 512)")
    ap.add_argument("--rob-k", type=int, default=4,
                    help="Noise-robustness perturbation count K "
                         "(protocol Set A uses K=32).")
    ap.add_argument("--eta", type=float, default=None,
                    help="η (clean-set scheduling probability, Set C dial). "
                         "Default None = auto-resolve 0.1 under --use-semi.")
    ap.add_argument("--correction-cond",
                    choices=["none", "x", "x_yhat", "yhat"], default="none",
                    help="Inputs of the correction map q(z | ẑ, ·).")
    ap.add_argument("--emit-only", action="store_true",
                    help="Skip training; load the saved ckpt bundle and only "
                         "re-emit protocol scores (e.g. plausibility reruns).")
    args = ap.parse_args()

    if args.rob_k < 1:
        raise ValueError(f"--rob-k must be >= 1 (got {args.rob_k})")

    loaded = load_embeddings(args.dataset)
    if isinstance(loaded, dict):
        # NoisyAG-News: real crowd noisy labels come from the file; the clean
        # oracle is ground_truth. No synthetic injection anywhere.
        X = loaded["X"]
        yc = loaded["y_clean"]      # ground truth oracle
        yn_all = loaded["y_noisy"]  # the variant's real crowd labels
        label_source = "real"
    else:
        X, yc = loaded
        # Synthetic noise: symmetric (uniform flips) or IDN (flip probability
        # scaled by proximity to other class centroids). Injected ONCE over the
        # full label array then subset by index, so every row carries the same
        # noisy label regardless of the clean-set size (row-consistent noise).
        if args.noise_type == "idn":
            yn_all = inject_noise_idn(yc, args.noise, args.seed,
                                      int(yc.max()) + 1, X=X)
        else:
            yn_all = inject_noise(yc, args.noise, args.seed, int(yc.max()) + 1)
        label_source = f"{args.noise_type}@{args.noise}"
    n_classes = int(yc.max()) + 1
    n = len(X)
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    rng = np.random.default_rng(split_seed)
    eval_idx = rng.choice(n, size=args.eval_slice, replace=False)
    rest = np.setdiff1d(np.arange(n), eval_idx)
    # Deterministic nested splits (fair clean-set ablation): ONE seeded
    # permutation of the pool; the clean set is always its PREFIX
    # clean_set[:N]. Larger clean sets strictly contain smaller ones — no
    # per-size re-randomization, so a size effect cannot be blamed on
    # drawing better rows by chance at some N. Validation is carved before
    # the clean prefix, hence invariant across N.
    order = rng.permutation(rest)
    n_val = min(args.val_size, len(order) // 2)
    val_idx = order[:n_val]
    pool = order[n_val:]
    N = resolve_clean_rows(args.clean_set_size, len(pool) // 2)
    cs_idx = pool[:N]
    tr_idx = pool[N:]

    Xe, yce = X[eval_idx], yc[eval_idx]
    Xtr, yctr = X[tr_idx], yc[tr_idx]
    Xcs, ycs = X[cs_idx], yc[cs_idx]
    Xv, ycv = X[val_idx], yc[val_idx]
    yne, yntr, ynv = yn_all[eval_idx], yn_all[tr_idx], yn_all[val_idx]
    print(f"[{args.dataset} {n_classes}cls, noise={label_source}] "
          f"splits: train={len(Xtr)} clean_set={len(Xcs)} val={len(Xv)} "
          f"eval={len(Xe)}; eval flip frac={(yne != yce).mean():.3f}")

    latent_dim = int(X.shape[1])  # embedding-space: latent dim == embed dim
    config = build_config(args.seed, args.epochs, args.beta, args.noise, args.batch_size,
                          args.clean_set_size, args.eval_slice, latent_dim, args.dataset,
                          n_classes, eta=args.eta,
                          correction_cond=args.correction_cond)
    # Tag the output dir with every dial: noise type keeps symmetric/IDN runs
    # apart; an explicit eta distinguishes the Set-C sweep (sweep-copy rule).
    config.output_dir = (f"/tmp/text_{args.dataset}_s{args.seed}_n{args.noise}"
                         f"_{args.noise_type}_e{args.epochs}_b{args.beta}_cs{args.clean_set_size}"
                         + (f"_eta{args.eta:g}" if args.eta is not None else ""))

    # Real Accelerator (never accel=None): device placement + mixed precision
    # come from `accelerate launch`; the trainer's DataLoaders are prepared so
    # clean-set/val batches land on device automatically.
    _ckpt_name = (f"lsnpc_{args.dataset}_s{args.seed}_n{args.noise}_{args.noise_type}"
                  f"_e{args.epochs}_b{args.beta}_cs{args.clean_set_size}"
                  + (f"_eta{args.eta:g}" if args.eta is not None else "")
                  + ("" if args.correction_cond == "none"
                     else f"_cc{args.correction_cond}") + ".pt")

    accel = Accelerator()
    device = str(accel.device)

    best_val_err_h = None
    fit_history = None
    if args.emit_only:
        # Load the trained bundle (saved by the original fit run) and skip
        # training: used to re-emit scores (e.g. posterior plausibility)
        # without refitting. Data prep above is identical to the fit run.
        _b = load_bundle(Path("results/ckpt/lsnpc/text") / _ckpt_name, device)
        ls = _b["model"].eval()
        print(f"[emit-only] loaded bundle {_ckpt_name} (training skipped)")
    else:
        enc = build_text_encoder(latent_dim=latent_dim)
        # The trainer reads these two from the config, and only the stage-1
        # helper sets them (experiments/lsnpc_stage1.py:453-456). Building
        # the trainer directly without mirroring them dies in
        # trainers/lsnpc.py:206 with AttributeError on lsnpc_z_vae_dim.
        config.lsnpc_z_vae_dim = int(getattr(enc, "latent_dim", latent_dim))
        config.lsnpc_eta = _resolve_lsnpc_eta(config, True)
        # TextEncoder.encode returns (embedding, None) -> the trainer's z_vae = embedding.
        trainer = LSNPCTrainer(model=None, config=config, device=device, accel=accel)
        trainer.train(vae=enc, X_train=Xtr, y_train_noisy=yntr,
                      X_clean_set=Xcs, y_clean_set=ycs, n_classes=n_classes,
                      X_val=Xv, y_val_clean=ycv, y_val_noisy=ynv,
                      use_semi=True)
        ls = trainer.model.eval()
        # NOTE: the trainer exposes best_val_error (property), not best_val_err_h.
        best_val_err_h = trainer.best_val_error
        fit_history = trainer.history
        print(f"val_err_h (best) = {best_val_err_h}")

    # Persist the trained LSNPC for re-use (Set E / emission mode).
    # Emit-only reruns keep the existing bundle untouched.
    if not args.emit_only:
        _out = save_bundle(
            Path("results/ckpt/lsnpc/text") / _ckpt_name, ls,
        arch={"build_kwargs": bundled_build_kwargs(
            config, x_dim=latent_dim, n_classes=n_classes)},
        splits={"eval_idx": eval_idx, "val_idx": val_idx,
                "cs_idx": cs_idx, "tr_idx": tr_idx},
        run=dict(dataset=args.dataset, seed=args.seed, noise=args.noise,
                 noise_type=args.noise_type, epochs=args.epochs, beta=args.beta,
                 clean_set_size=args.clean_set_size, eval_slice=args.eval_slice,
                 val_size=args.val_size, batch_size=args.batch_size,
                 split_seed=int(split_seed)),
        val_err_h=trainer.best_val_error)
        print(f"[ckpt] LSNPC bundle saved: {_out}")

    # ── Correction inference + CF protocol scores on the eval slice ──
    # All score computations live in experiments/protocol.py — shared by
    # every pipeline so the axes are computed identically.
    bs = int(config.batch_size); M = int(getattr(config, "iw_samples", 5))
    proto = compute_protocol_scores(
        ls, Xe, yce, yne, device, batch_size=bs, M=M, seed=args.seed,
        k_perturb=args.rob_k)
    corr = proto["corr"]
    probs = proto["probs"]
    rows = proto["rows"]
    valid_at_f = {k: v for k, v in proto["scores"].items()
                  if k.startswith("valid_at")}
    mis = proto["mis"]
    succ = proto["succ"]

    # Deployed-frame re-emission (EMIT_FRAMES=1, paper sec:deploy): the
    # self-referenced variants of the two label-referenced axes, computed on
    # the same rows and in the same pass as the instrument values above.
    emit_frame_report(
        config.output_dir, ls, Xe, yce, yne, device, batch_size=bs, M=M,
        seed=args.seed, k_perturb=args.rob_k,
        stored_auroc={r[0]: r[1] for r in rows})

    print(f"\n=== text SST-2: mis n={mis.sum()} validity_mis={succ[mis].mean():.3f} "
          f"({succ[mis].sum()}/{mis.sum()}) ===")
    print(f"{'feature':24s} {'repAUROC':>8s} {'repAP':>7s} {'valid_mean':>11s} "
          f"{'invalid_mean':>13s} {'dmgAUROC':>8s} {'dmgAP':>7s}")
    for name, a, ap, vm, im, da, dap in rows:
        print(f"{name:24s} {a:8.3f} {ap:7.3f} {vm:11.4f} {im:13.4f} "
              f"{da:8.3f} {dap:7.3f}")
    print(f"transitions: {proto['transitions']}  "
          f"(WR=repair, RW=right->wrong damage, RR=kept, WW=still-wrong)")

    # Per-query score export (transition figure): unique per-run file.
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"[out] {config.output_dir}")
    _npz_path = os.path.join(config.output_dir,
                             f"per_query_scores_s{args.seed}_b{args.beta}.npz")
    np.savez_compressed(
        _npz_path,
        transition=(np.where(mis & succ, "WR",
                     np.where(mis & ~succ, "WW",
                     np.where(~mis & ~succ, "RW", "RR")))),
        **{f"score__{k}": np.asarray(v, dtype=np.float64)
           for k, v in proto["scores"].items()
           if k in SCORE_KEYS and v is not None})

    res = {
        "dataset": args.dataset,
        "beta": args.beta, "noise": args.noise, "noise_type": args.noise_type,
        "seed": args.seed, "eta": args.eta,
        "clean_set_size": args.clean_set_size,
        "n_train": len(Xtr), "n_val": len(Xv), "n_eval": len(Xe),
        "val_size": args.val_size, "eval_slice": args.eval_slice,
        "rob_k": args.rob_k,
        "mis_frac": float(mis.mean()), "validity_mis": float(succ[mis].mean()),
        "val_err_h": best_val_err_h,
        "flip_frac": float((corr != yne).mean()),
        "cf_auroc": {name: rows[i][1] for i, (name, *_r) in enumerate(rows)},
        "cf_ap": {name: rows[i][2] for i, (name, *_r) in enumerate(rows)},
        "damage_auroc": proto["damage_auroc"],
        "damage_ap": proto["damage_ap"],
        "transitions": proto["transitions"],
        "valid_at": {k: float(v.mean()) for k, v in valid_at_f.items()},
        # Corrected-result quality on the eval slice vs clean labels.
        "corr_acc": float((corr == yce).mean()),
        "corr_f1": multiclass_f1(corr, yce),
        "corr_precision": multiclass_precision(corr, yce),
        "corr_recall": multiclass_recall(corr, yce),
        # Noisy-label baseline quality on the eval slice vs clean labels.
        "noisy_acc": float((yne == yce).mean()),
        "noisy_f1": multiclass_f1(yne, yce),
        "noisy_precision": multiclass_precision(yne, yce),
        "noisy_recall": multiclass_recall(yne, yce),
        # Per-epoch training record: loss, KL terms, clean-set loss, val
        # errors, and wall-clock seconds per epoch (Phase A + Phase B).
        "history": (
            {"epoch": fit_history["epoch"], "loss": fit_history["loss"],
             "kl_zhat": fit_history["kl_zhat"], "kl_z": fit_history["kl_z"],
             "clean_set_loss": fit_history["clean_set_loss"],
             "val_err_h": fit_history["val_err_h"],
             "val_err_noisy": fit_history["val_err_noisy"],
             "epoch_time_s": fit_history["epoch_time_s"],
             "total_train_time_s": sum(fit_history["epoch_time_s"]),
            } if fit_history is not None else None),
    }
    print(json.dumps(res, indent=2))
    out = Path(f"{config.output_dir}/result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
