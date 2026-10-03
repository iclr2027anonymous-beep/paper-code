#!/usr/bin/env python3
"""Set F -- pool enlargement on the CIFAR real-noise testbeds (frozen RN50).

The same question the AG News pool study asks, on the image side: does the
score threshold select pool rows that are worth admitting into the clean pool
of a round-1 semi-supervised retrain? Per seed:

  1. splits are REPLICATED exactly as ``image_lsnpc.main`` builds them (same
     ``--split-seed``, eval/val carve then one seeded permutation of the rest,
     clean set = its prefix), so this study indexes the same rows the banked
     Set E cells do -- the corrector is the banked Set E frozen-RN50 corrector
     for the seed, loaded rather than refit;
  2. full pass over the candidate pool: the M1 (minimality) gate axis AND the
     corrected label for every row;
  3. the admission threshold is calibrated on the disjoint validation split at
     >=95% precision, as pre-registered;
  4. four size-matched round-1 retrains differ only in which candidate rows
     join the clean pool -- V0 (none), Gated (threshold passes), Unguarded and
     Random (uniform draws of the same size, independent streams) -- and every
     admitted row carries the label the corrector emitted for it, never the
     ground truth;
  5. each round-1 model is evaluated on the held-out eval slice against the
     true labels (``valid_clean``).

Ground truth is a measurement instrument here: it scores the gate's precision
and the final accuracy. It never enters training.

Usage:
    accelerate launch --mixed_precision bf16 -m scripts.e6_cifar_pool_expansion \
        --seeds 42 43 44 --testbeds cifar10n cifar100n
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from experiments.lsnpc_stage1 import resolve_clean_rows
from experiments.protocol import (
    MINIMALITY_KEY,
    compute_stream_scores,
    multiclass_f1,
)
from models.rn50_features import IdentityFeatureEncoder
from scripts import e_common
from scripts.e6_downstream_eval import head_eval
from scripts.image_lsnpc import build_config, load_image_pair
from scripts.lsnpc_ckpt import load_bundle
from trainers.lsnpc import LSNPCTrainer

CONFIG_NAMES = ("V0", "Gated", "Unguarded", "Random")
# Quarter volume is dropped deliberately: the selective set is held to three
# methods (the calibrated gate, the top-decile cut, the top-half cut), each
# with its volume-matched random baseline. Keep this list and the testbeds'
# cell sets consistent -- adding a volume here re-fits every seed.
DEFAULT_COVERAGES = (0.1, 0.5)
CELLS_PATH = Path("results/set_f/e6_cifar_cells.json")
CKPT_DIR = Path("results/ckpt/lsnpc/image")
TESTBEDS = {"cifar10n": "worse", "cifar100n": "fine"}


# ── splits (verbatim replication of image_lsnpc.main) ───────────────────

def build_splits(n: int, split_seed: int, eval_slice: int, val_size: int,
                 clean_set_size: int) -> dict:
    """Row indices for eval / val / clean set / train, matching the E cells."""
    rng = np.random.default_rng(split_seed)
    ev_n = min(eval_slice, n // 4)
    eval_idx = rng.choice(n, size=ev_n, replace=False)
    rest = np.setdiff1d(np.arange(n), eval_idx)
    order = rng.permutation(rest)
    n_val = min(val_size, len(order) // 2)
    val_idx = order[:n_val]
    pool = order[n_val:]
    n_cs = resolve_clean_rows(clean_set_size, len(pool) // 2)
    cs_idx = pool[:n_cs]
    tr_idx = pool[n_cs:]
    return {"eval": eval_idx, "val": val_idx, "cs": cs_idx, "tr": tr_idx}


def _corrector_path(ds: str, view: str, seed: int, epochs: int, beta: float,
                    cs: int) -> Path:
    return CKPT_DIR / (f"lsnpc_{ds}_s{seed}_e{epochs}_b{beta:g}"
                       f"_cs{cs}_{view}.pt")


# ── protocol helpers ────────────────────────────────────────────────────

def _score(corr_model, X, y_clean, y_noisy, device, bs, M, seed, tag):
    """M1 gate axis + the corrected label for every row, in one full pass."""
    out = compute_stream_scores(
        corr_model, X, y_noisy, device, batch_size=bs, M=M, seed=seed,
        which=(MINIMALITY_KEY,), y_clean=y_clean)
    m1 = np.asarray(out["scores"][MINIMALITY_KEY], dtype=np.float64)
    succ = np.asarray(out["succ"], dtype=bool)
    corr = np.asarray(out["corr"]).astype(np.int64)
    print(f"[E6C] {tag}: n={len(X)} corrected-valid={succ.mean():.3f} "
          f"edits={int(out['edited'].sum())}", flush=True)
    return m1, succ, corr


def _with_weights(configs, weights):
    """Extend the configuration list over admission weights.

    Weight 1.0 keeps the plain name, so the existing cells remain the baseline
    and nothing is re-fitted. Any other weight appends ``@w<value>`` and reuses
    the same admission mask: ``Gated@w0.9`` is exactly the Gated selection with
    its corrected labels down-weighted inside the round-1 clean branch. V0
    admits nothing, so it is never duplicated.
    """
    out = []
    for name, mask in configs:
        if name == "V0":
            out.append((name, mask))
            continue
        for w in weights:
            out.append((name, mask) if float(w) == 1.0
                       else (f"{name}@w{float(w):g}", mask))
    return out


def _train_round1(corr_bundle_vae, F_train, y_train_served, Fc1, yc1,
                  Fc1_served, F_val, ycv, ynv, n_classes, config, accel, device,
                  admit_weight: float = 1.0, n_base: int | None = None):
    """Round-1 semi-supervised retrain on the frozen-feature path.

    ``admit_weight`` is the confidence the clean branch gives the admitted
    rows' corrected labels; those are the last ``len(Fc1) - n_base`` rows of the
    clean set, and the known-clean rows ahead of them keep weight 1.0. The
    default 1.0 reproduces the hard-label expansion exactly.
    """
    config._clean_set_yhat = np.asarray(Fc1_served).astype(np.int64)
    if admit_weight != 1.0:
        nb = len(Fc1) if n_base is None else int(n_base)
        na = len(Fc1) - nb
        if na < 0:
            raise ValueError(f"n_base={nb} exceeds clean set size {len(Fc1)}")
        config._clean_set_weight = np.concatenate([
            np.ones(nb, dtype=np.float64),
            np.full(na, float(admit_weight), dtype=np.float64)])
    trainer = LSNPCTrainer(model=None, config=config, device=device, accel=accel)
    trainer.train(
        vae=corr_bundle_vae, X_train=F_train, y_train_noisy=y_train_served,
        X_clean_set=Fc1, y_clean_set=yc1, n_classes=n_classes,
        X_val=F_val, y_val_clean=ycv, y_val_noisy=ynv,
        use_semi=True,
    )
    return trainer


# ── per-seed driver ─────────────────────────────────────────────────────

def run_seed(ds: str, view: str, seed: int, args, cells: dict) -> None:
    device = str(Accelerator().device)
    X, y_noisy_all, y_clean_all, _y_test, _m, _s = load_image_pair(
        ds, noise_type=view, noise_rate=0.0)
    n_classes = int(max(y_clean_all.max(), y_noisy_all.max())) + 1
    n = len(X)
    sp = build_splits(n, args.split_seed, args.eval_slice, args.val_size,
                      args.clean_set_size)
    print(f"\n=== [E6C] {ds} view={view} seed={seed} "
          f"(pool={n}, train={len(sp['tr'])} cal={len(sp['val'])} "
          f"cs={len(sp['cs'])} eval={len(sp['eval'])}) ===", flush=True)

    ckpt = _corrector_path(ds, view, seed, args.epochs, args.beta,
                           args.clean_set_size)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"needs the banked Set E frozen-RN50 corrector {ckpt}")
    bundle = load_bundle(ckpt, device)
    # The bundle records the splits the corrector was fitted with: index those
    # rows exactly rather than trusting a re-derivation, and check the
    # re-derivation against them so a drift in the split recipe shows up loudly.
    s = bundle["splits"]
    ev, va, cs, tr = (np.asarray(s["eval_idx"]), np.asarray(s["val_idx"]),
                      np.asarray(s["cs_idx"]), np.asarray(s["tr_idx"]))
    for name, got, want in (("eval", sp["eval"], ev), ("val", sp["val"], va),
                            ("cs", sp["cs"], cs), ("tr", sp["tr"], tr)):
        if not np.array_equal(got, want):
            print(f"[E6C] WARNING: re-derived {name} split != bundle split "
                  f"({len(got)} vs {len(want)} rows) — using the bundle's",
                  flush=True)
    print(f"[E6C] splits from bundle: train={len(tr)} cal={len(va)} "
          f"cs={len(cs)} eval={len(ev)} | run={bundle['run']}", flush=True)

    F_all = e_common.rn50_features_cached(X, ds, view, "pool", device)
    del X
    gc.collect()
    Fe, Fv, Fcs, Ftr = F_all[ev], F_all[va], F_all[cs], F_all[tr]

    ls0 = bundle["model"].eval()
    vae = bundle.get("vae") or IdentityFeatureEncoder()
    bs = int(args.batch_size)
    _tmp = build_config(seed, args.epochs, args.beta, args.batch_size,
                        args.clean_set_size, args.eval_slice,
                        int(F_all.shape[1]), ds, n_classes)
    M = int(getattr(_tmp, "iw_samples", 5))
    del _tmp

    m1_pool, succ_pool, corr_pool = _score(
        ls0, Ftr, y_clean_all[tr], y_noisy_all[tr], device, bs, M, seed,
        f"{ds} seed={seed} candidate pool")
    m1_val, succ_val, _ = _score(
        ls0, Fv, y_clean_all[va], y_noisy_all[va], device, bs, M, seed,
        f"{ds} seed={seed} validation split")

    curve, best = e_common.gate_curve(m1_val, succ_val)
    print("[E6C] gate curve: " + " | ".join(
        (f"prec>={r['target']:.2f}: cov={r['coverage']:.3f} "
         f"prec={r['precision']:.3f}") if r["threshold"] is not None
        else f"prec>={r['target']:.2f}: INFEASIBLE" for r in curve), flush=True)
    reach = [r for r in curve if r["threshold"] is not None]
    if reach:
        gate_target = reach[0]["target"]
        thresh, cov, prec = (reach[0]["threshold"], reach[0]["coverage"],
                             reach[0]["precision"])
        gated = m1_pool >= thresh
        print(f"[E6C] gate target={gate_target:.2f} threshold={thresh:.4f} "
              f"coverage={cov:.3f} precision={prec:.3f} "
              f"-> N_gated={int(gated.sum())}", flush=True)
    else:
        gate_target, thresh, cov, prec = None, None, None, None
        gated = np.zeros(len(Ftr), dtype=bool)
        print("[E6C] gate infeasible at every declared target -> every "
              "configuration "
              "degenerate to V0", flush=True)
    n_gated = int(gated.sum())

    del ls0
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    configs = _with_weights(
        e_common.admission_configurations(m1_pool, gated, n_gated, seed, args.coverages),
        args.admit_weights)
    if args.only_configs:
        keep_c = {c.strip() for c in args.only_configs.split(",")}
        configs = [c for c in configs if c[0] in keep_c]
    e_common.dump_pool_context(ds, seed, corr=corr_pool, m1=m1_pool, succ=succ_pool,
                 y_clean=y_clean_all[tr], y_noisy=y_noisy_all[tr],
                 configs=configs, eval_idx=ev, val_idx=va, cs_idx=cs,
                 tr_idx=tr, view=view)
    for cfg, admit in configs:
        _wtag = cfg.partition("@w")[2]
        admit_weight = float(_wtag) if _wtag else 1.0
        # coverage tag, if this is a 'c@<coverage>' / 'f@<coverage>' configuration:
    # read it off the
        # part of the name before any '@w' weight suffix, so 'c@0.5@w0.9' still
        # reports coverage 0.5 rather than choking on 'w0.9'.
        _qtag = cfg.partition("@w")[0].partition("@")[2]
        ckey = f"{ds}:{seed}:t{gate_target}:{cfg}:d{int(args.downstream)}"
        if ckey in cells:
            print(f"[E6C] {ckey} cached — skip", flush=True)
            continue
        if cfg == "V0":
            Fc1, yc1, yc1_served = Fcs, y_clean_all[cs], y_clean_all[cs]
            desc = "clean set only"
            admit = np.zeros(len(Ftr), dtype=bool)
        else:
            desc = f"{cfg} N={int(admit.sum())}"
            Xa = Ftr[admit]
            ya = yc_served = corr_pool[admit]
            Fc1 = np.concatenate([Fcs, Xa], axis=0)
            yc1 = np.concatenate([y_clean_all[cs], ya], axis=0)
            yc1_served = np.concatenate([y_clean_all[cs], yc_served], axis=0)
        n_adm = 0 if cfg == "V0" else int(admit.sum())
        adm_prec = (float(succ_pool[admit].mean()) if n_adm else None)
        # An admitted row is a row whose label we now trust: it MOVES out of the
        # noisy training pool and into the clean pool. Leaving it in both
        # branches trains the same row against contradictory labels (its noisy
        # label in the semi-supervised pass, its corrected label in the clean
        # pass), which is what the first version of this harness did.
        keep = ~admit
        Ftr_kept, ytr_kept = Ftr[keep], y_noisy_all[tr][keep]
        n_removed = int((~keep).sum())
        adm_acc = (float((corr_pool[admit] == y_clean_all[tr][admit]).mean())
                   if n_adm else None)
        print(f"[E6C] {ds} seed={seed} config={cfg} pool={desc} "
              f"clean_set={len(Fc1)}", flush=True)

        cfg2 = build_config(seed, args.epochs, args.beta, args.batch_size,
                            args.clean_set_size, args.eval_slice,
                            int(Ftr.shape[1]), ds, n_classes)
        cfg2.output_dir = (f"/tmp/e6c_{ds}_s{seed}_b{args.beta:g}"
                           f"_e{args.epochs}_{cfg.lower().replace('@', '_')}")
        a2 = Accelerator()
        t1 = _train_round1(vae, Ftr_kept, ytr_kept, Fc1, yc1, yc1_served,
                           Fv, y_clean_all[va], y_noisy_all[va], n_classes,
                           cfg2, a2, str(a2.device),
                           admit_weight=admit_weight, n_base=len(cs))
        out = compute_stream_scores(
            t1.model.eval(), Fe, y_noisy_all[ev], str(a2.device),
            batch_size=bs, M=M, seed=seed, which=(MINIMALITY_KEY,),
            y_clean=y_clean_all[ev])
        corr = np.asarray(out["corr"])
        yce = y_clean_all[ev]
        # Step 3 of the pipeline: correct the DATA with this model, then train
        # the downstream task on the corrected dataset. The round-1 model is
        # applied to the whole training pool (not just the rows it was given as
        # clean supervision), so the downstream head sees the dataset this
        # configuration actually produces.
        if args.downstream:
            assert len(Ftr) == len(y_noisy_all[tr]), (
                f"emission inputs/labels misaligned: {len(Ftr)} vs "
                f"{len(y_noisy_all[tr])}")
            r1 = compute_stream_scores(
                t1.model.eval(), Ftr, y_noisy_all[tr], str(a2.device),
                batch_size=bs, M=M, seed=seed, which=(MINIMALITY_KEY,),
                y_clean=y_clean_all[tr])
            corr_r1 = np.asarray(r1["corr"]).astype(np.int64)
            ds_res = head_eval(Ftr, corr_r1, Fe, yce, n_classes, seed,
                               str(a2.device))
            print(f"[E6C] {cfg} seed={seed}: downstream_acc="
                  f"{ds_res['acc']:.4f} (corrected-pool labels "
                  f"{(corr_r1 == y_clean_all[tr]).mean():.3f} valid)",                  flush=True)
        entry = {
            "dataset": ds, "view": view, "seed": seed, "config": cfg,
            "pool_labels": "corrector-full-pass",
            "admitted_label_role": ("supervision = conditioning input = the "
                                    "corrector's label (admitted rows join a "
                                    "genuinely clean pool here)"),
            "admitted_label_acc": adm_acc,
            "n_admitted": n_adm,
            "n_train_removed": n_removed,
            "admitted_precision": adm_prec,
            "coverage": (float(_qtag) if _qtag else None),
            "admit_weight": admit_weight,
            "gate_target": gate_target, "gate_curve": curve,
            "gate_best_precision": best["precision"],
            "gate_best_coverage": best["coverage"],
            "gate_threshold": thresh, "gate_coverage": cov,
            "gate_precision": prec,
            "valid_clean": float((corr == yce).mean()),
            "corr_f1": float(multiclass_f1(corr, yce)),
            **({"downstream_acc": ds_res["acc"],
                "downstream_f1": ds_res["f1"],
                "r1_pool_label_valid": float(
                    (corr_r1 == y_clean_all[tr]).mean())}
               if args.downstream else {}),
            "noisy_f1": float(multiclass_f1(y_noisy_all[ev], yce)),
        }
        cells[ckey] = entry
        CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CELLS_PATH.write_text(json.dumps(cells, indent=2))
        print(f"[E6C] {cfg} seed={seed}: valid_clean={entry['valid_clean']:.4f}",
              flush=True)
        del t1, Fc1, yc1, yc1_served
        gc.collect()
        if str(a2.device).startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--testbeds", nargs="+", default=["cifar10n", "cifar100n"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--clean-set-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=2500)
    ap.add_argument("--eval-slice", type=int, default=3000)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--downstream", type=int, default=1,
                    help="train the downstream head on the corrected dataset")
    ap.add_argument("--admit-weights", type=float, nargs="+", default=[1.0],
                    help="Confidence weights on admitted rows' corrected labels; "
                         "1.0 = hard labels (baseline), e.g. 0.9 down-weights")
    ap.add_argument("--only-configs", default=None,
                    help="comma list: run only these configurations")
    ap.add_argument("--coverages", nargs="+", type=float, default=list(DEFAULT_COVERAGES),
                    help="admission fractions for the selectivity sweep")
    args = ap.parse_args()

    cells = json.loads(CELLS_PATH.read_text()) if CELLS_PATH.exists() else {}
    for ds in args.testbeds:
        view = TESTBEDS[ds]
        for seed in args.seeds:
            run_seed(ds, view, seed, args, cells)

    summary = {"study": "set_f_cifar", "cells": len(cells), "configs": {}}
    names = sorted({v["config"] for v in cells.values()},
                   key=lambda n: (n not in CONFIG_NAMES, n))
    for cfg in names:
        vals = [v["valid_clean"] for v in cells.values()
                if v["config"] == cfg]
        if vals:
            summary["configs"][cfg] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals), "per_cell": vals}
    for a, b, label in (("Gated", "Unguarded", "gated - unguarded (calibrated)"),
                        ("c@0.1", "f@0.1", "c@0.1 - f@0.1"),
                        ("c@0.25", "f@0.25", "c@0.25 - f@0.25"),
                        ("c@0.5", "f@0.5", "c@0.5 - f@0.5")):
        av = summary["configs"].get(a, {}).get("mean")
        bv = summary["configs"].get(b, {}).get("mean")
        if av is not None and bv is not None:
            summary[f"{a}_minus_{b}"] = av - bv
            print(f"=== [E6C] {label} = {av - bv:+.4f} ===")
    out = Path("results/set_f/e6_cifar_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(json.dumps(summary["configs"], indent=2))


if __name__ == "__main__":
    main()
