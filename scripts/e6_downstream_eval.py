#!/usr/bin/env python3
"""Set F downstream evaluation: what does each admission configuration's dataset buy?

The Set F harnesses score a configuration by the *corrector's* own accuracy on a
held-out slice (``valid_clean``). That is the wrong yardstick for the question
the study asks: a corrector that leaves labels alone scores well there while
producing worse training data. The deliverable is a classifier trained on the
corrected data, so this stage trains exactly that and compares test accuracy.

For each (testbed, seed, configuration) the dataset is the label set the
configuration defines -- the rows it did NOT admit keep their observed noisy
labels, the clean set keeps its true labels, and the admitted rows carry the
corrector's emitted labels -- and a frozen-feature MLP head (the Set E recipe:
hidden 256 x 3, dropout 0.1, AdamW 1e-3, early stop on a stratified 10% internal
split) is trained on it and evaluated on the held-out clean slice.

Reference points recorded alongside: the same head on the all-noisy pool (the
status quo with nothing corrected) and, where the pool's true labels are
trustworthy, on the oracle labels.

Contexts come from ``results/set_f/pool_ctx/<tag>_s<seed>.npz``, written by the
harnesses so this stage never refits a corrector. CIFAR is rebuilt from its
banked Set E corrector when no context file exists (its corrector is loaded, not
fitted, so the rebuild is exact).

Usage:
    python -m scripts.e6_downstream_eval --testbeds cifar10n noisyag_symmetric
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.protocol import compute_stream_scores, multiclass_f1
from scripts import e_common
from scripts.image_lsnpc import load_image_pair
from scripts.lsnpc_ckpt import load_bundle
from scripts.retrain_e_streams import fit_mlp
from scripts.text_lsnpc_sst2 import load_embeddings

FEAT = Path("results/set_e/_features")
CTX = Path("results/set_f/pool_ctx")
OUT = Path("results/set_f")
DOP_VIEW = "symmetric"


# ── feature sources ─────────────────────────────────────────────────────

def _cifar_features(ds: str, view: str) -> np.ndarray:
    p = FEAT / f"feat_{ds}_{view}_rn50_pool_sz32.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing cached features {p}")
    return np.load(p)


def _dopanim_features() -> tuple[np.ndarray, np.ndarray]:
    p = FEAT / f"feat_dopanim_{DOP_VIEW}_rn50_pool_sz224.npy"
    t = FEAT / f"feat_dopanim_{DOP_VIEW}_rn50_test_sz224.npy"
    if not p.exists() or not t.exists():
        raise FileNotFoundError(
            f"missing dopanim feature caches ({p.name}, {t.name})")
    return np.load(p), np.load(t)


# ── contexts ────────────────────────────────────────────────────────────

def context_cifar(ds: str, view: str, seed: int) -> dict:
    """Rebuild the CIFAR context from the banked Set E corrector (exact)."""
    # Lazy import: e6_cifar_pool_expansion imports this module at top level.
    from scripts.e6_cifar_pool_expansion import (
        MINIMALITY_KEY,
        _corrector_path,
    )

    X, y_noisy, y_clean, _yt, _m, _s = load_image_pair(
        ds, noise_type=view, noise_rate=0.0)
    del X
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _corrector_path(ds, view, seed, 10, 0.5, 2000)
    bundle = load_bundle(ckpt, device)
    s = bundle["splits"]
    ev = np.asarray(s["eval_idx"]); va = np.asarray(s["val_idx"])
    cs = np.asarray(s["cs_idx"]); tr = np.asarray(s["tr_idx"])
    F = _cifar_features(ds, view)
    Ftr, Fv = F[tr], F[va]
    model = bundle["model"].eval()
    bs, M = 256, 5

    def score(Fx, yc, yn):
        out = compute_stream_scores(model, Fx, yn, device, batch_size=bs, M=M,
                                    seed=seed, which=(MINIMALITY_KEY,),
                                    y_clean=yc)
        return (np.asarray(out["scores"][MINIMALITY_KEY], dtype=np.float64),
                np.asarray(out["succ"], dtype=bool),
                np.asarray(out["corr"]).astype(np.int64))

    m1_pool, succ_pool, corr_pool = score(Ftr, y_clean[tr], y_noisy[tr])
    m1_val, succ_val, _ = score(Fv, y_clean[va], y_noisy[va])
    curve, _best = e_common.gate_curve(m1_val, succ_val)
    reach = [r for r in curve if r["threshold"] is not None]
    gated = (m1_pool >= reach[0]["threshold"]) if reach else np.zeros(len(tr), bool)
    configs = e_common.admission_configurations(
        m1_pool, gated, int(gated.sum()), seed, (0.1, 0.25, 0.5))

    def build(mask):
        keep = ~mask
        X = np.concatenate([F[tr][keep], F[cs], F[tr][mask]], axis=0)
        y = np.concatenate([y_noisy[tr][keep], y_clean[cs], corr_pool[mask]])
        return X, y.astype(np.int64)

    return {"corr": corr_pool, "m1": m1_pool, "succ": succ_pool,
            "y_clean": y_clean[tr], "y_noisy": y_noisy[tr],
            "eval_idx": ev, "val_idx": va, "cs_idx": cs, "tr_idx": tr,
            "configs": configs, "build": build, "feature_all": F,
            "feat_eval": F[ev], "y_eval": y_clean[ev],
            "trusted_pool_labels": True, "n_classes": int(y_clean.max()) + 1}


def context_from_npz(tag: str, seed: int, features, feat_eval, y_eval,
                     n_classes: int, trusted_pool_labels: bool,
                     raw_order: np.ndarray | None = None) -> dict:
    p = CTX / f"{tag}_s{seed}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"missing context {p} — the harness writes it during a run; for "
            f"datasets already run without it, re-run the harness step")
    z = np.load(p)
    d = {k: z[k] for k in z.files if not k.startswith("mask__")}
    configs = [(k[len("mask__"):], z[k].astype(bool))
               for k in z.files if k.startswith("mask__")]
    # dopanim stores the pool permutation: harness row i is raw pool row perm[val_size + i]
    if raw_order is not None:
        off = int(d["val_size"])
        d["tr_raw"] = np.asarray(raw_order)[off:]
        d["ev_raw"] = np.asarray(d["eval_idx"])
    return {"corr": d["corr"], "m1": d["m1"], "succ": d["succ"],
            "y_clean": d["y_clean"], "y_noisy": d["y_noisy"],
            "cs_idx": d.get("cs_idx"), "eval_idx": d.get("eval_idx"),
            "configs": configs, "features": features, "feat_eval": feat_eval,
            "y_eval": y_eval, "trusted_pool_labels": trusted_pool_labels,
            "n_classes": n_classes, "raw": d}


# ── library entry point used by the harnesses (design A) ────────────────

def head_eval(features: np.ndarray, labels: np.ndarray,
              feats_eval: np.ndarray, y_eval: np.ndarray, n_classes: int,
              seed: int, device: str | None = None) -> dict:
    """Train the Set E-recipe head on a corrected dataset and score it.

    ``features``/``labels`` are the dataset the stage produced (the round-1
    corrector's labels for the pool), ``feats_eval``/``y_eval`` the held-out
    clean slice. One head seed (= the corrector seed); the head recipe is
    ``retrain_e_streams.fit_mlp`` unmodified, so Set F's downstream numbers sit
    on the same footing as Set E's.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = fit_mlp(np.asarray(features), np.asarray(labels).astype(np.int64),
                  seed, n_classes, int(np.asarray(features).shape[1]), device)
    with torch.no_grad():
        pred = net(torch.as_tensor(np.asarray(feats_eval), dtype=torch.float32,
                                   device=device)).argmax(dim=1).cpu().numpy()
    return {"acc": float((pred == np.asarray(y_eval)).mean()),
            "f1": float(multiclass_f1(pred, np.asarray(y_eval)))}


# ── the experiment ──────────────────────────────────────────────────────

def dataset_for(ctx: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The label set a configuration defines: rows it did not admit keep their
    observed noisy labels, the clean set keeps its true labels, the admitted
    rows carry the corrector's emitted labels."""
    return ctx["build"](mask)


def evaluate(ctx: dict, head_seeds: list[int], device: str) -> dict:
    out = {}
    for name, mask in ctx["configs"]:
        Xtr, ytr = dataset_for(ctx, mask)
        accs, f1s = [], []
        for hs in head_seeds:
            net = fit_mlp(Xtr, ytr, hs, ctx["n_classes"], Xtr.shape[1], device)
            with torch.no_grad():
                pred = net(torch.as_tensor(ctx["feat_eval"], dtype=torch.float32,
                                           device=device)).argmax(dim=1).cpu().numpy()
            accs.append(float((pred == ctx["y_eval"]).mean()))
            f1s.append(float(multiclass_f1(pred, ctx["y_eval"])))
        out[name] = {"n_train": int(len(Xtr)), "n_admitted": int(mask.sum()),
                     "acc": accs, "acc_mean": float(np.mean(accs)),
                     "acc_std": float(np.std(accs)),
                     "f1_mean": float(np.mean(f1s))}
        print(f"    {name:9s} n_train={len(Xtr):6d} acc={np.mean(accs):.4f}"
              f" (+-{np.std(accs):.4f})", flush=True)
    # reference: nothing corrected at all (noisy pool only, no clean set)
    if ctx.get("trusted_pool_labels"):
        Xn = ctx["feature_all"][ctx["tr_idx"]]
        yn, on = ctx["y_noisy"], ctx["y_clean"]
        for tag, Xa, ya in (("all_noisy", Xn, yn), ("oracle", Xn, on)):
            accs = []
            for hs in head_seeds:
                net = fit_mlp(Xa, ya.astype(np.int64), hs, ctx["n_classes"],
                              Xa.shape[1], device)
                with torch.no_grad():
                    pred = net(torch.as_tensor(ctx["feat_eval"],
                                               dtype=torch.float32,
                                               device=device)).argmax(dim=1).cpu().numpy()
                accs.append(float((pred == ctx["y_eval"]).mean()))
            out[tag] = {"n_train": int(len(Xa)), "acc_mean": float(np.mean(accs)),
                        "acc_std": float(np.std(accs)), "acc": accs}
            print(f"    {tag:9s} n_train={len(Xa):6d} acc={np.mean(accs):.4f}",
                  flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbeds", nargs="+", default=["cifar10n"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--head-seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for tb in args.testbeds:
        recs = {}
        for seed in args.seeds:
            print(f"\n=== [F-down] {tb} seed={seed} ===", flush=True)
            if tb.startswith("cifar"):
                view = "worse" if tb == "cifar10n" else "fine"
                ctx = context_cifar(tb, view, seed)
            elif tb.startswith("noisyag"):
                X, yc = load_embeddings("ag_news")
                z = np.load(CTX / f"{tb}_s{seed}.npz")
                ev = np.asarray(z["eval_idx"])
                tr = np.asarray(z["tr_idx"])
                cs = np.asarray(z["cs_idx"])
                ctx = context_from_npz(tb, seed, X, X[ev], yc[ev],
                                       int(yc.max()) + 1, True)
                ctx["tr_idx"], ctx["feature_all"] = tr, X

                def build(mask, X=X, tr=tr, cs=cs, z=z, yc=yc):
                    keep = ~mask
                    Xa = np.concatenate([X[tr][keep], X[cs], X[tr][mask]], axis=0)
                    ya = np.concatenate([z["y_noisy"][keep], yc[cs],
                                         z["corr"][mask]])
                    return Xa, ya.astype(np.int64)
                ctx["build"] = build
            else:
                Fp, Ft = _dopanim_features()
                z = np.load(CTX / f"{tb}_s{seed}.npz")
                ev = np.asarray(z["eval_idx"])
                cs = np.asarray(z["cs_idx"])
                y_test = e_common.load_dopanim_view(224)[4]
                ctx = context_from_npz(tb, seed, Fp, Ft[ev], y_test[ev],
                                       int(y_test.max()) + 1, False,
                                       raw_order=z["perm"])
                off = int(z["val_size"]) + int(z.get("cal_size", 0))
                tr_raw = np.asarray(z["perm"])[off:]

                def build(mask, Fp=Fp, Ft=Ft, tr_raw=tr_raw, cs=cs, z=z,
                          y_test=y_test):
                    keep = ~mask
                    Xa = np.concatenate([Fp[tr_raw[keep]], Ft[cs],
                                         Fp[tr_raw[mask]]], axis=0)
                    ya = np.concatenate([z["y_noisy"][keep], y_test[cs],
                                         z["corr"][mask]])
                    return Xa, ya.astype(np.int64)
                ctx["build"] = build
            recs[seed] = evaluate(ctx, args.head_seeds, device)
        OUT.mkdir(parents=True, exist_ok=True)
        summary = {"testbed": tb, "head_seeds": args.head_seeds,
                   "cells": recs, "means": {}}
        names = sorted({n for r in recs.values() for n in r})
        for n in names:
            v = [r[n]["acc_mean"] for r in recs.values() if n in r]
            summary["means"][n] = {"mean": float(np.mean(v)),
                                   "std": float(np.std(v)), "n": len(v)}
        v0 = summary["means"].get("V0", {}).get("mean")
        if v0 is not None:
            for n, m in summary["means"].items():
                m["vs_V0"] = m["mean"] - v0
        p = OUT / f"e6_downstream_{tb}.json"
        p.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {p}")
        for n, m in summary["means"].items():
            print(f"  {n:9s} mean={m['mean']:.4f} +-{m['std']:.4f}"
                  f"  vs V0 {m.get('vs_V0', float('nan')):+.4f}")


if __name__ == "__main__":
    main()
