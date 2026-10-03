#!/usr/bin/env python3
"""Set E — end-to-end downstream retrain on corrected streams (ENG-3).

Fits the six pre-registered downstream training configurations on ONE
emitted label stream (the retrain pool the corrector was fit to clean) and
evaluates each on the clean test:

    (a) original noisy labels          (baseline: train on the stream as-is)
    (b) blanket-corrected              (keep every correction)
    (c) gated-corrected @ coverage     (keep top-coverage edits by the pre-committed
                                        axis; plausibility floor alpha=0.1 on
                                        image; text axis = minimality_self,
                                        image axis = robustness_self)
    (d) clean-oracle ceiling           (testbeds with a per-row clean oracle
                                        in the pool: NoisyAG, CIFAR-N; NOT
                                        dopanim — pre-registered E2)
    (e) in-training robust baseline      (Co-teaching two-head small-loss over
                                        the noisy labels; GCE single-network
                                        fallback selected per cell BEFORE
                                        results — the shell pre-commits one)
    (f) random-rejection @ coverage control   (same emissions as (c), edits accepted
                                        uniformly at random at the same count)

Labels are the ONLY difference: head architecture, init rule and schedule are
identical across configurations; the encoder is frozen per modality (text:
mpnet embeddings; image embedding mode: frozen RN50 features; image pixel
mode: the corrector's own image-encoder features), exactly as E3/E4 specify.

Head:  [frozen features] -> MLP[hidden x layers, ReLU, dropout] -> logits,
trained with AdamW + early stopping on an internal stratified 10% split of
the retrain stream (never the clean test). Seeds {42,43,44} are paired
across configurations; corrector + emissions are shared, so the delta
variance is downstream-only.

Usage (one call per cell = testbed x corrector-seed):
    python -m scripts.retrain_e_streams \
        --tag e_text_noisyag_worst_cs42 \
        --emissions results/set_e/e_text_noisyag_worst_cs42/emissions_cs42.npz \
        --configs "a b c d e f" --coverages "0.1 0.25 0.5" --seeds "42 43 44 45 46"
Writes results/set_e/cells/<tag>.json (per-config acc/F1 per seed) plus the
per-seed detail lines under results/set_e/cells/<tag>_detail.json.
"""
from __future__ import annotations

from utils.paths import project_path

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.protocol import multiclass_f1
from scripts import e_common
from scripts.lsnpc_ckpt import load_bundle
from utils.batching import place_loader_local, tensor_loader

RESULTS = Path(project_path('results/set_e'))

MODALITY_BY_TESTBED = {}
for _t in e_common.TEXT_TESTBEDS:
    MODALITY_BY_TESTBED[_t] = "text"
for _t in e_common.IMAGE_TESTBEDS:
    MODALITY_BY_TESTBED[_t] = "image"

GATE_AXIS = {"text": "minimality(1-minf)_self",
             "image": "noise_robustness_self"}
PLAUS_FLOOR_ALPHA = 0.10   # image only, pre-committed (Set B / plan E3)


# ── head ─────────────────────────────────────────────────────────────────


def build_head(in_dim: int, n_classes: int, hidden: int, layers: int,
               dropout: float = 0.1) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [n_classes]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
            if dropout > 0.0:
                mods.append(nn.Dropout(dropout))
    return nn.Sequential(*mods)


def _stratified_split(X, y, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    tr, va = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nv = max(1, int(round(frac * len(idx))))
        va.append(idx[:nv])
        tr.append(idx[nv:])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    return tr, va


def _as_tensors(X, y, device, bs):
    """Shuffled batcher over device-resident inputs (seed-pinned order)."""
    return tensor_loader(
        torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(y), dtype=torch.long, device=device),
        batch_size=bs, shuffle=True,
        generator=torch.Generator().manual_seed(0))


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int, n_classes: int,
            in_dim: int, device: str, hidden: int = 256, layers: int = 3,
            epochs: int = 40, lr: float = 1e-3, batch: int = 512,
            dropout: float = 0.1, patience: int = 8,
            verbose: bool = False) -> nn.Module:
    """Frozen-feature MLP head, fresh init, early stop on a stratified 10%
    internal split of the retrain stream (never the clean test)."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = _stratified_split(X, y, 0.1, seed)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    net = build_head(in_dim, n_classes, hidden, layers, dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_va, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in _as_tensors(Xtr, ytr, device, batch):
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va_err = float(crit(net(torch.as_tensor(Xva, dtype=torch.float32,
                                                    device=device)),
                                torch.as_tensor(yva, dtype=torch.long,
                                                device=device)).item())
        if va_err < best_va - 1e-5:
            best_va, best_state, bad = va_err, \
                {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.eval()


def predict_proba(net: nn.Module, X: np.ndarray, device: str,
                  batch: int = 512) -> np.ndarray:
    dl = place_loader_local(
        tensor_loader(torch.as_tensor(X, dtype=torch.float32),
                      batch_size=batch, shuffle=False),
        device)
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            out.append(torch.softmax(net(xb), dim=-1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def build_linear(in_dim: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, n_classes))


# Asymmetric-loss settings for the asl family, chosen for this study.
# Reference points: the ASL paper's own defaults are gamma_neg=4, gamma_pos=1,
# clip=0.05 (Alibaba-MIIL/ASL, src/loss_functions/losses.py); the earlier
# in-project ASL runs used gamma_neg=1, gamma_pos=0.5, clip=0.01. This study
# uses gamma_neg=1, gamma_pos=0.25, clip=0.05.
ASL_GAMMA_NEG = 1.0
ASL_GAMMA_POS = 0.25
ASL_MARGIN = 0.05


def _asl_loss(gamma_neg: float = ASL_GAMMA_NEG,
              gamma_pos: float = ASL_GAMMA_POS, m: float = ASL_MARGIN):
    """Multi-class asymmetric loss: per-class sigmoid focal with asymmetric
    focusing and probabilistic margin shifting (Ben-Baruch et al., 2020),
    generalised to multi-class one-hot targets. The defaults are this study's
    settings; see ASL_GAMMA_NEG/POS/MARGIN above for provenance."""

    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.one_hot(target, num_classes=logits.shape[1])
        y = y.float().to(logits.device)
        x = logits - m * (1.0 - y)          # margin shift on negatives
        xt = x + (2.0 * m) * y * (1.0 - y)  # keep positives unshifted
        p = torch.sigmoid(xt)
        pt_pos = p * y + (1.0 - p) * (1.0 - y)
        gamma = gamma_pos * y + gamma_neg * (1.0 - y)
        term = torch.pow(1.0 - pt_pos, gamma) * torch.log(pt_pos.clamp(min=1e-8))
        return -(term.sum(-1)).mean()

    return loss


def fit_linear(X: np.ndarray, y: np.ndarray, seed: int, n_classes: int,
               in_dim: int, device: str, epochs: int = 60,
               lr: float = 1e-2, batch: int = 512) -> nn.Module:
    """Single-layer logistic head on the same frozen features: the cheapest
    possible downstream model (sensitivity floor for the MLP result)."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = _stratified_split(X, y, 0.1, seed)
    net = build_linear(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_va, best_state, bad = float("inf"), None, 0
    Xva_t = torch.as_tensor(X[va], dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(y[va], dtype=torch.long, device=device)
    for ep in range(epochs):
        net.train()
        for xb, yb in _as_tensors(X[tr], y[tr], device, batch):
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va_loss = float(crit(net(Xva_t), yva_t))
        if va_loss < best_va - 1e-5:
            best_va, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def fit_asl(X: np.ndarray, y: np.ndarray, seed: int, n_classes: int,
            in_dim: int, device: str, hidden: int = 256, layers: int = 3,
            epochs: int = 40, lr: float = 1e-3, batch: int = 512,
            dropout: float = 0.1) -> nn.Module:
    """Same MLP head trained with the asymmetric loss (ASL) instead of
    cross-entropy: a loss-level robustness comparator."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = _stratified_split(X, y, 0.1, seed)
    net = build_head(in_dim, n_classes, hidden, layers, dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = _asl_loss()
    best_va, best_state, bad = float("inf"), None, 0
    Xva_t = torch.as_tensor(X[va], dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(y[va], dtype=torch.long, device=device)
    for ep in range(epochs):
        net.train()
        for xb, yb in _as_tensors(X[tr], y[tr], device, batch):
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va_loss = float(crit(net(Xva_t), yva_t))
        if va_loss < best_va - 1e-5:
            best_va, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def fit_coteach(X: np.ndarray, y: np.ndarray, seed: int, n_classes: int,
                in_dim: int, mis_frac: float, device: str,
                hidden: int = 256, layers: int = 3, epochs: int = 40,
                lr: float = 1e-3, batch: int = 512, dropout: float = 0.2,
                warmup_frac: float = 0.1, verbose: bool = False):
    """Co-teaching: two heads over the same frozen features; each head trains
    the OTHER on the small-loss fraction (1 - R(t)) of its batch; the forget
    rate R(t) warms linearly over the first `warmup_frac` of steps to
    R_final >= mis_frac (pre-committed schedule). Predictions average the two
    heads (report-head rule). Returns (model_a, model_b)."""
    torch.manual_seed(seed); np.random.seed(seed)
    r_final = float(np.clip(mis_frac + 0.05, 0.2, 0.5))
    head_a = build_head(in_dim, n_classes, hidden, layers, dropout).to(device)
    head_b = build_head(in_dim, n_classes, hidden, layers, dropout).to(device)
    heads = [head_a, head_b]
    opts = [torch.optim.AdamW(h.parameters(), lr=lr) for h in heads]
    crit = nn.CrossEntropyLoss(reduction="none")
    n_train = len(X)
    dl = _as_tensors(X, y, device, batch)
    n_batches = int(np.ceil(n_train / batch))
    warm_steps = max(1, int(warmup_frac * n_batches * epochs))
    step = 0
    for ep in range(epochs):
        for xb, yb in dl:
            step += 1
            r_t = (min(step / warm_steps, 1.0) * r_final
                   if step < warm_steps else r_final)
            keep = max(2, int((1.0 - r_t) * len(xb)))
            with torch.no_grad():
                losses = [crit(h(xb), yb) for h in heads]
                sel = [torch.topk(losses[i], keep, largest=False)[1]
                       for i in range(2)]
            # Standard Co-teaching cross-update: head i learns the small-loss
            # subset chosen by the partner head (1 - i), evaluated under head
            # i's own loss — each head's gradient updates ITS OWN parameters.
            loss_i = [crit(heads[i](xb[sel[1 - i]]), yb[sel[1 - i]]).mean()
                      for i in range(2)]
            for i in range(2):
                opts[i].zero_grad()
                loss_i[i].backward()
                opts[i].step()
    return head_a.eval(), head_b.eval()


def fit_gce(X: np.ndarray, y: np.ndarray, seed: int, n_classes: int,
            in_dim: int, device: str, q: float = 0.7,
            hidden: int = 256, layers: int = 3, epochs: int = 40,
            lr: float = 1e-3, batch: int = 512, dropout: float = 0.1,
            verbose: bool = False) -> nn.Module:
    """GCE (Zhang & Sabuncu) single-network fallback: loss = (1 - p_y^q)/q."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_head(in_dim, n_classes, hidden, layers, dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for ep in range(epochs):
        net.train()
        for xb, yb in _as_tensors(X, y, device, batch):
            opt.zero_grad()
            p = torch.softmax(net(xb), dim=-1)
            py = p.gather(1, yb.unsqueeze(1)).squeeze(1)
            loss = ((1.0 - torch.clamp(py, min=1e-8) ** q) / q).mean()
            loss.backward()
            opt.step()
    return net.eval()


# ── stream assembly (labels only differ) ────────────────────────────────


def assemble_streams(em: dict, pool: np.ndarray, coverages, modality: str,
                     clean_ceiling: bool, seeds) -> dict:
    y_noisy = em["y_noisy"][pool]
    corr = em["corr"][pool]
    edited = em["edited"][pool]
    has_clean = "y_clean" in em
    y_clean = em["y_clean"][pool] if has_clean else None

    def score_col(key: str) -> np.ndarray:
        return em[f"score__{key}"][pool]

    axis = GATE_AXIS[modality]
    s = score_col(axis)
    plaus = score_col("plausibility(density)") if modality == "image" else None

    # Candidate set = edited rows (ĉ != stream label), exactly the B gate
    # population (WR ∪ RW in the oracle view; oracle-free here).
    cand = np.where(edited)[0]
    out = {"a": y_noisy, "b": corr,
           "mis_frac_pool": (float((y_noisy != y_clean).mean())
                             if y_clean is not None else None),
           "n_pool": int(len(y_noisy)), "n_edited": int(edited.sum())}
    if clean_ceiling:
        if y_clean is None:
            raise SystemExit("clean ceiling requested but no y_clean in pool")
        out["d"] = y_clean

    for cov in coverages:
        n_admit = int(np.ceil(cov * len(cand)))
        tau = float(np.quantile(s[cand], 1.0 - cov)) if len(cand) else np.nan
        pi0 = None
        admit = np.zeros(len(y_noisy), dtype=bool)
        if len(cand):
            rank = s[cand]
            if modality == "image":
                # plausibility floor: alpha-quantile of the edited val/heldout
                # candidates when available, else the pool edited stream.
                hc = em["is_holdout"] & em["edited"]
                if hc.sum() >= 30:
                    pi0 = float(np.quantile(
                        em["score__plausibility(density)"][hc], PLAUS_FLOOR_ALPHA))
                else:
                    pi0 = float(np.quantile(plaus[edited], PLAUS_FLOOR_ALPHA))
                keep_ix = np.argsort(-rank, kind="stable")[:n_admit]
                ok_plaus = plaus[cand] >= pi0
                # top-coverage by axis; a plaus-rejected edit drops out (no fill-in:
                # rank cut is pre-committed and one-sided).
                top = np.zeros(len(cand), dtype=bool)
                top[keep_ix] = True
                admit_c = top & ok_plaus
            else:
                admit_c = np.zeros(len(cand), dtype=bool)
                admit_c[np.argsort(-rank, kind="stable")[:n_admit]] = True
            admit[cand] = admit_c
        out[f"c@{cov}"] = np.where(admit, corr, y_noisy)
        out[f"gate@{cov}"] = dict(n_edits=int(len(cand)),
                                n_admit=int(admit.sum()),
                                tau=tau, pi_floor=pi0,
                                coverage=float(admit.sum() / max(1, len(cand))))
        # (f) random rejection at equal count, per downstream seed.
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)
            n = out[f"gate@{cov}"]["n_admit"]
            rcand = rng.choice(cand, size=n, replace=False) if n else np.zeros(0, int)
            rl = y_noisy.copy()
            rl[rcand] = corr[rcand]
            out[f"f@{cov}@{sd}"] = rl
    return out


# ── metrics ──────────────────────────────────────────────────────────────


def evaluate(net, X_test: np.ndarray, y_test: np.ndarray, device: str) -> dict:
    proba = predict_proba(net, X_test, device)
    pred = proba.argmax(-1)
    return {"acc": float((pred == y_test).mean()),
            "f1": float(multiclass_f1(pred, y_test)),
            "pred": pred}


def mean_sd(x) -> dict:
    a = np.asarray(x, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std()) if len(a) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emissions", type=Path, default=None)
    ap.add_argument("--testbed", required=True,
                    choices=sorted(MODALITY_BY_TESTBED))
    ap.add_argument("--configs", default="a b c d e f")
    ap.add_argument("--coverages", default="0.1")
    ap.add_argument("--seeds", default="42 43 44 45 46")
    ap.add_argument("--device", default=None)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--head-layers", type=int, default=3)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--co-method", choices=("coteach", "gce"), default="coteach")
    ap.add_argument("--models", default="mlp",
                    help="comma list of downstream model families: mlp, linear, asl")
    ap.add_argument("--no-clean-ceiling", action="store_true",
                    help="dopanim: no clean train labels -> config (d) skipped")
    ap.add_argument("--a-only", action="store_true",
                    help="dopanim abstention path: run downstream config (a) "
                         "on the original noisy labels only (no corrector, "
                         "no emission) plus the gateability accounting")
    ap.add_argument("--gate-file", type=Path, default=None,
                    help="C-c gateability verdict json (dopanim abstention "
                         "accounting)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    modality = MODALITY_BY_TESTBED[args.testbed]
    coverages = [float(x) for x in args.coverages.split()]
    seeds = [int(x) for x in args.seeds.split()]
    configs = args.configs.split()
    clean_ceiling = not args.no_clean_ceiling
    if args.a_only:
        if args.testbed != "dopanim":
            raise SystemExit("--a-only is the dopanim abstention path only")
        return dopanim_a_only(args, device, seeds)

    if args.emissions is None:
        raise SystemExit("--emissions is required unless --a-only")
    # ── emissions ────────────────────────────────────────────────────
    em = dict(np.load(args.emissions, allow_pickle=True))
    pool = np.where(em["is_pool"].astype(bool))[0]
    streams = assemble_streams(em, pool, coverages, modality, clean_ceiling, seeds)

    # Held-out label-level diagnostics (out-of-sample corrector rows).
    held = np.where(em["is_holdout"].astype(bool))[0]
    label_net = None
    if "y_clean" in em and len(held):
        yn, corr, yc = em["y_noisy"][held], em["corr"][held], em["y_clean"][held]
        mis = yn != yc
        succ = corr == yc
        wr = int((mis & succ).sum()); rw = int((~mis & ~succ).sum())
        label_net = dict(n=len(held), WR=wr, RW=rw, net=wr - rw,
                         validity_mis=float(succ[mis].mean()),
                         corr_acc=float((corr == yc).mean()),
                         noisy_acc=float((yn == yc).mean()),
                         mis_frac=float(mis.mean()))

    # ── features of the retrain pool + clean test ───────────────────
    row_global = em["row_global"][pool].astype(np.int64)
    feat, test_feat, y_test = _load_features(
        args.emissions, em, args.tag, modality, args.testbed, row_global,
        device)
    n_classes = int(max(int(em["y_noisy"].max()), int(y_test.max()))) + 1
    in_dim = int(feat.shape[1])

    # ── per-config fits (labels only differ) ─────────────────────────
    cell = {
        "tag": args.tag, "modality": modality, "testbed": args.testbed,
        "corr_seed": int(str(args.tag).split("_cs")[-1]),
        "img_size": None,
        "co_method": args.co_method,
        "head": {"hidden": args.head_hidden, "layers": args.head_layers,
                 "dropout": args.head_dropout, "epochs": args.epochs,
                 "lr": args.lr, "batch": args.batch},
        "clean_ceiling": clean_ceiling,
        "stream": {"n_pool": streams["n_pool"],
                   "mis_frac_pool": streams["mis_frac_pool"],
                   "n_edited": streams["n_edited"],
                   "heldout": label_net},
        "gates": {cov: streams[f"gate@{cov}"] for cov in coverages},
        "configs": {},
        "models": {},
    }
    # Extend mode: preserve previously computed families when re-running with a
    # larger --models list (e.g. adding linear/asl after an mlp-only cell), and
    # preserve previously computed gate records when re-running with a wider
    # --coverages grid (adding coverage 0.7/0.8/0.9 to the {0.1,0.25,0.5} grid). Without the
    # "gates" merge the cell keeps only the coverages of THIS invocation and the
    # banked taus/pi_floor/coverage of the earlier grid are lost.
    _prev = RESULTS / "cells" / f"{args.tag}.json"
    if _prev.exists():
        _old = json.loads(_prev.read_text())
        for _k in ("configs", "models", "gates"):
            if isinstance(_old.get(_k), dict):
                cell[_k].update(_old[_k])

    families = [f.strip() for f in str(args.models).split(",") if f.strip()]
    for fam in families:
        if fam not in ("mlp", "linear", "asl"):
            raise SystemExit(f"unknown downstream model family: {fam}")
        cell["models"].setdefault(
            fam, {"head": dict(cell["head"]) if fam != "linear" else
                  {"type": "linear", "epochs": args.epochs, "lr": 1e-2,
                   "batch": args.batch}, "configs": {}})

    def _fit_family(family, yl, sd):
        if family == "linear":
            return fit_linear(feat, yl, sd, n_classes, in_dim, device,
                              epochs=args.epochs)
        if family == "asl":
            return fit_asl(feat, yl, sd, n_classes, in_dim, device,
                           hidden=args.head_hidden, layers=args.head_layers,
                           epochs=args.epochs, lr=args.lr, batch=args.batch,
                           dropout=args.head_dropout)
        return fit_mlp(feat, yl, sd, n_classes, in_dim, device,
                       args.head_hidden, args.head_layers,
                       args.epochs, args.lr, args.batch, args.head_dropout)

    _current_family = ["mlp"]

    def run_cfg(name, ylabels, co_mis=None):
        # (e) is itself a different model class (Co-teaching/GCE): it belongs to
        # the mlp family only; other families share the same streams.
        fams = families if name != "e" else [f for f in families if f == "mlp"]
        for fam in fams:
            _current_family[0] = fam
            _run_cfg_one(name, ylabels, co_mis)

    def _run_cfg_one(name, ylabels, co_mis=None):
        accs, f1s = [], []
        for sd in seeds:
            if name == "e" and args.co_method == "coteach":
                ha, hb = fit_coteach(feat, ylabels, sd, n_classes, in_dim,
                                     co_mis, device, args.head_hidden,
                                     args.head_layers, args.epochs, args.lr,
                                     args.batch)
                def _proba2(xx):
                    return (predict_proba(ha, xx, device)
                            + predict_proba(hb, xx, device)) / 2
                pred = _proba2(test_feat).argmax(-1)
                acc = float((pred == y_test).mean())
                f1 = float(multiclass_f1(pred, y_test))
            else:
                if name == "e":
                    net = fit_gce(feat, ylabels, sd, n_classes, in_dim, device,
                                  hidden=args.head_hidden,
                                  layers=args.head_layers, epochs=args.epochs,
                                  lr=args.lr, batch=args.batch)
                else:
                    net = _fit_family(_current_family[0], ylabels, sd)
                ev = evaluate(net, test_feat, y_test, device)
                acc, f1 = ev["acc"], ev["f1"]
            accs.append(acc); f1s.append(f1)
        fam0 = _current_family[0]
        cell["models"].setdefault(fam0, {"configs": {}})
        cell["models"][fam0]["configs"][name] = {
            "seeds": seeds, "acc": accs, "f1": f1s,
            "acc_sum": mean_sd(accs), "f1_sum": mean_sd(f1s)}
        if fam0 == "mlp":
            cell["configs"][name] = {"seeds": seeds, "acc": accs, "f1": f1s,
                                     "acc_sum": mean_sd(accs),
                                     "f1_sum": mean_sd(f1s)}
        print(f"[retrain] cfg {name} [{'+'.join(families)}]: acc "
              f"{np.mean(accs):.4f}+-{np.std(accs):.4f}  f1 "
              f"{np.mean(f1s):.4f}+-{np.std(f1s):.4f}", flush=True)

    # (e) uses the noisy labels with the Co-teaching small-loss schedule.
    co_mis = streams["mis_frac_pool"]
    for name in configs:
        if name == "a":
            run_cfg("a", streams["a"])
        elif name == "b":
            run_cfg("b", streams["b"])
        elif name == "d":
            if "d" in streams:
                run_cfg("d", streams["d"])
            else:
                print("[retrain] skipping (d): no clean ceiling (dopanim)")
        elif name == "e":
            if co_mis is None:
                raise SystemExit("cfg (e) needs the stream mis_frac — run on "
                                 "a testbed whose pool carries clean labels")
            run_cfg("e", streams["a"], co_mis=co_mis)
        elif name in ("c", "f"):
            def _run_f(cov):
                # (f) is the primary-model control — the cell documents it as
                # "computed for this model only" — so it must always be an mlp
                # fit. `_current_family[0]` is whatever family the preceding
                # run_cfg("c@<coverage>") finished on, so under --models mlp,linear,asl
                # the control would be silently recorded as an ASL fit under a
                # key the table reads as the MLP control.
                if "mlp" not in families:
                    return
                if f"f@{cov}@{seeds[0]}" not in streams:
                    return
                accs, f1s = [], []
                for sd in seeds:
                    yl = streams[f"f@{cov}@{sd}"]
                    net = _fit_family("mlp", yl, sd)
                    ev = evaluate(net, test_feat, y_test, device)
                    accs.append(ev["acc"]); f1s.append(ev["f1"])
                cell["configs"][f"f@{cov}"] = {
                    "seeds": seeds, "acc": accs, "f1": f1s,
                    "acc_sum": mean_sd(accs), "f1_sum": mean_sd(f1s)}
                print(f"[retrain] cfg f@{cov}: acc "
                      f"{np.mean(accs):.4f} f1 {np.mean(f1s):.4f}")
            for cov in coverages:
                if name == "c":
                    run_cfg(f"c@{cov}", streams[f"c@{cov}"])
                    _run_f(cov)
                elif "c" not in configs:
                    _run_f(cov)
    if modality == "image":
        # The emission sidecar always exists (scripts.emit_lsnpc_stream
        # writes it); a missing/corrupt sidecar must fail loudly, not
        # silently drop the resolution from the run-tag record.
        _info = json.loads(Path(args.emissions).with_suffix(".json").read_text())
        cell["img_size"] = _info.get("img_size")
    # The driver owns the artifact namespace: the unsupervised run writes
    # under results/set_e_unsup, so the cells dir is overridable rather
    # than pinned to the semi-supervised RESULTS tree.
    cells_dir = Path(os.environ.get("SET_E_CELLS_DIR", RESULTS / "cells"))
    cells_dir.mkdir(parents=True, exist_ok=True)
    (cells_dir / f"{args.tag}.json").write_text(json.dumps(cell, indent=2))
    print(f"[retrain] wrote {cells_dir / (args.tag + '.json')}")



def dopanim_a_only(args, device, seeds) -> None:
    """dopanim E6 abstention path: no 224px C-c cell passes the gateability
    gate, so configuration (b)/(c) would collapse onto (a). Pre-registered
    default until a gateable cell appears: run (a) on the original noisy
    labels with a frozen RN50-feature head over the FULL annotated pool,
    evaluate on the final clean test carve (2500 rows of the 4500-row clean
    test pool that the would-be corrector clean set does NOT touch), and
    record the C-c gateability accounting verbatim.
    """
    Xp, y_noisy, y_clean, Xt, yt = e_common.load_dopanim_view()
    corr_seed = int(str(args.tag).split("_cs")[-1])
    # Index-fixed dopanim carve (one seeded permutation shared by every
    # corrector seed); clean reference 2000 -> final test = the remaining
    # 2500 rows of the same permutation.
    cs_idx, test_idx = e_common.dopanim_test_carve(
        corr_seed, eval_slice=3000, val_size=2500, split_seed=42,
        allow_large_eval=True)
    Fp = e_common.rn50_features_cached(Xp, "dopanim", "symmetric", "pool",
                                       device, args.batch)
    Ft_all = e_common.rn50_features_cached(Xt, "dopanim", "symmetric", "test",
                                           device, args.batch)
    Ft = Ft_all[test_idx]
    yte = np.asarray(yt)[test_idx]
    in_dim = int(Fp.shape[1])
    n_classes = int(max(y_noisy.max(), yte.max())) + 1
    mis_frac = float((y_noisy != y_clean).mean())
    fams = [f.strip() for f in str(getattr(args, "models", "mlp")).split(",") if f.strip()]
    def _fit_dop(fam, sd):
        if fam == "linear":
            return fit_linear(Fp, y_noisy, sd, n_classes, in_dim, device,
                              epochs=args.epochs)
        if fam == "asl":
            return fit_asl(Fp, y_noisy, sd, n_classes, in_dim, device,
                           hidden=args.head_hidden, layers=args.head_layers,
                           epochs=args.epochs, lr=args.lr, batch=args.batch,
                           dropout=args.head_dropout)
        return fit_mlp(Fp, y_noisy, sd, n_classes, in_dim, device,
                       args.head_hidden, args.head_layers, args.epochs,
                       args.lr, args.batch, args.head_dropout)
    models = {}
    legacy = {}
    for fam in fams:
        accs, f1s = [], []
        for sd in seeds:
            net = _fit_dop(fam, sd)
            ev = evaluate(net, Ft, yte, device)
            accs.append(ev["acc"]); f1s.append(ev["f1"])
        models[fam] = {"configs": {"a": {"seeds": seeds, "acc": accs, "f1": f1s,
                                          "acc_sum": mean_sd(accs),
                                          "f1_sum": mean_sd(f1s)}}}
        if fam == "mlp":
            legacy = {"seeds": seeds, "acc": accs, "f1": f1s,
                      "acc_sum": mean_sd(accs), "f1_sum": mean_sd(f1s)}
    gate = {}
    if args.gate_file and args.gate_file.exists():
        gate = json.loads(args.gate_file.read_text())
    cell = {
        "tag": args.tag, "modality": "image", "testbed": "dopanim",
        "corr_seed": corr_seed, "co_method": "n/a (abstention)",
        "head": {"hidden": args.head_hidden, "layers": args.head_layers,
                 "dropout": args.head_dropout, "epochs": args.epochs,
                 "lr": args.lr, "batch": args.batch},
        "clean_ceiling": False,
        "dopanim_abstention": True,
        "img_size": int(Xp.shape[-1]),
        "gate_accounting": gate,
        "carve": {"clean_set_rows": int(len(cs_idx)),
                  "test_rows": int(len(test_idx)),
                  "n_clean_test_pool": int(len(Xt)),
                  "split_seed": 42},
        "stream": {"n_pool": int(len(Xp)),
                   "mis_frac_pool": mis_frac,
                   "n_edited": None,
                   "heldout": None},
        "gates": {},
        "configs": {"a": legacy} if legacy else {},
        "models": models,
    }
    _prev = RESULTS / "cells" / f"{args.tag}.json"
    if _prev.exists():
        _oldm = (json.loads(_prev.read_text()).get("models") or {})
        for _f, _blk in _oldm.items():
            cell["models"].setdefault(_f, _blk)
    # The driver owns the artifact namespace: the unsupervised run writes
    # under results/set_e_unsup, so the cells dir is overridable rather
    # than pinned to the semi-supervised RESULTS tree.
    cells_dir = Path(os.environ.get("SET_E_CELLS_DIR", RESULTS / "cells"))
    cells_dir.mkdir(parents=True, exist_ok=True)
    (cells_dir / f"{args.tag}.json").write_text(json.dumps(cell, indent=2))
    print(f"[retrain] dopanim abstention: cfg (a) acc "
          f"{np.mean(accs):.4f}+-{np.std(accs):.4f} f1 "
          f"{np.mean(f1s):.4f}; carve test n={len(test_idx)} "
          f"(clean set n={len(cs_idx)} disclosed)")
    print(f"[retrain] wrote {cells_dir / (args.tag + '.json')}")


def _load_features(em_path: Path, em: dict, tag: str, modality: str,
                   testbed: str, row_global: np.ndarray, device: str,
                   batch: int = 512):
    """Return (pool features, test features, y_test) aligned to row_global."""
    # The emission sidecar json records the exact data view used.
    info = json.loads(em_path.with_suffix(".json").read_text())
    ckpt = info["ckpt"]
    pixel = bool(info.get("pixel", False))
    img_size = info.get("img_size")
    run = info["run"]

    if modality == "text":
        X_all, yc, yn = e_common.load_text_view(testbed)
        Fp = np.ascontiguousarray(X_all[row_global])
        Xte, yte = e_common.load_text_test()
        return Fp, Xte, yte
    if pixel:
        # The corrector's own image-encoder deterministic features (frozen at
        # downstream time). load the bundle and featurize pool + test with the
        # same encoder weights the corrector used.
        bundle = load_bundle(ckpt, device)
        model = bundle["model"]
        if testbed == "cifar10n":
            view = "worse"
        elif testbed == "cifar100n":
            view = "fine"
        else:
            view = "symmetric"
        if testbed in ("cifar10n", "cifar100n"):
            X_all, ynv, ycv, Xte_raw, yte = e_common.load_cifar_view(
                testbed, view)
            test_ix = np.arange(len(Xte_raw))
        elif testbed == "animal10n":
            # Pixel-mode Animal-10N: the corrector is a Swin, so its own
            # image encoder is the featurizer, exactly as for the other
            # pixel testbeds. The pool is the annotated train split and the
            # test is the held-out clean rows. The dopanim carve in the
            # branch below is dopanim-specific and would index the wrong
            # array (a 10.5k pool against 47.5k row ids).
            X_all, ynv, _ycv, Xte_raw, yte = e_common.load_animal10n_view()
            test_ix = np.arange(len(Xte_raw))
        else:
            X_all, ynv, ycv, Xte_raw, yte = e_common.load_dopanim_view()
            # dopanim: final test = the 2500 clean test-pool rows that the
            # corrector clean set (2000 rows of the same seeded permutation)
            # does not touch. Replay the fit-time RNG (index-fixed split seed
            # 42 across corrector seeds).
            corr_seed = int(str(tag).split("_cs")[-1])
            ev_sl = int(run.get("eval_slice", 1000))
            _, test_ix = e_common.dopanim_test_carve(
                corr_seed, eval_slice=ev_sl,
                val_size=int(run.get("val_size", 2500)),
                split_seed=int(run.get("split_seed", 42)),
                allow_large_eval=bool(run.get("allow_large_eval", False)))
        enc = model.noisy_encoder.image_encoder
        Fp = _encode_repr(enc, X_all[row_global], device)
        Fte = _encode_repr(enc, Xte_raw[test_ix], device)
        return Fp, Fte, np.asarray(yte)[test_ix]
    # image embedding mode: frozen RN50 features, cached.
    if testbed == "animal10n":
        # Same E2 shape as dopanim (annotations on the train split only, clean
        # set from the clean test pool) but these correctors are frozen-RN50
        # embedding mode, so the cached-feature path is the supported one. The
        # test is the held-out clean rows the corrector never saw; there is no
        # clean train oracle, so ycv is None and no ceiling is computable.
        X_all, ynv, _ycv, Xte_raw, yte = e_common.load_animal10n_view()
        Fp = e_common.rn50_features_cached(
            X_all, testbed, "native", "pool", device, batch)[row_global]
        Fte = e_common.rn50_features_cached(
            Xte_raw, testbed, "native", "test", device, batch)
        return Fp, Fte, np.asarray(yte)
    if testbed == "cifar10n":
        view, mean, std = "worse", e_common.CIFAR10_MEAN, e_common.CIFAR10_STD
    elif testbed == "cifar100n":
        view, mean, std = "fine", e_common.CIFAR100_MEAN, e_common.CIFAR100_STD
    else:
        raise SystemExit("dopanim E correctors are pixel; embedding-mode "
                         "retrain is not supported for dopanim")
    X_all, ynv, ycv, Xte_raw, yte = e_common.load_cifar_view(testbed, view)
    Fp = e_common.rn50_features_cached(X_all, testbed, view, "pool",
                                       device, batch)[row_global]
    Fte = e_common.rn50_features_cached(Xte_raw, testbed, view, "test",
                                        device, batch)
    return Fp, Fte, yte


def _encode_repr(enc: nn.Module, X_norm_nchw: np.ndarray, device: str,
                 batch: int = 512) -> np.ndarray:
    dl = place_loader_local(
        tensor_loader(torch.as_tensor(X_norm_nchw, dtype=torch.float32),
                      batch_size=batch, shuffle=False),
        device)
    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            mu, _ = enc(xb)
            outs.append(mu.float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


if __name__ == "__main__":
    main()
