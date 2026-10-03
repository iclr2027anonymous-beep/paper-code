#!/usr/bin/env python3
"""E6 in the destructive regime: pool expansion on dopanim (real crowd noise).

The AG News study of ``scripts/e6_pool_expansion.py`` measures the null where
correction is *constructive* (injected noise at rate 0.2 on text, a setting in
which the unguarded stream already beats the threshold). The regime the paper's
own account says the threshold is for -- severe, destructive real crowd noise
with a held-out clean evaluation -- has never been run through the E6 design.
dopanim supplies it: 10,484 rows annotated by 3-10 humans (single-annotator
argmax carries ~31.5% real noise, measured 0.313 on the eval carve) and 4,500
guaranteed-clean test rows, so both the pool and the evaluation are real.

Protocol (same four size-matched configurations as the AG News study):

    V0        round-1 clean set = carved clean set only (reference)
    Gated     admit candidate rows whose M1 score passes the calibrated gate
    Unguarded admit a RANDOM sample of the same size (volume control)
    Random    same-size random sample (auxiliary reference)

Per seed:
  1. carve the clean set (2,000) and the evaluation slice (1,000) from the
     4,500-row clean TEST pool via ``dopanim_test_carve(split_seed=42)`` --
     index-fixed across corrector seeds, as in the Set E dopanim recipe;
  2. hold out a gate-calibration split (2,000 annotated rows, drawn with the
     same split seed) from round-0 training: the M1 gate is calibrated on rows
     the round-0 corrector never saw, which is the point of the AG News design;
  3. round-0: semi-supervised LSNPC on the remaining annotated rows with the
     clean set as supervision (frozen Swin @224, the gateable dopanim recipe);
  4. score the candidate pool with the M1 axis only (the F gate is M1-only) and
     calibrate the threshold at >=95% precision on the held-out split;
  5. four round-1 trainings, each evaluated by the accuracy of its corrected
     stream against the held-out clean labels.

Deviation from the Set E dopanim recipe, declared: the 2,000-row calibration
split is held out of round-0 training. The E recipe trains on the full annotated
pool, which leaves no out-of-sample rows on which to calibrate an admission
gate; the held-out split is what makes the gate honest.

Usage:
    accelerate launch --mixed_precision bf16 -m scripts.e6_destructive_pool_expansion \
        --seeds 42 43 44 --epochs 10 --beta 0.5
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator

from experiments.protocol import (
    MINIMALITY_KEY,
    compute_stream_scores,
    multiclass_f1,
)
from scripts import e_common
from scripts.e6_downstream_eval import head_eval
from scripts.image_lsnpc import build_config
from trainers.lsnpc import _encode_vae_latents
from trainers.plausibility_vae import ckpt_path
from trainers.plausibility_vae import get_or_train as get_or_train_vae

CONFIG_NAMES = ("V0", "Gated", "Unguarded", "Random")
# per-cell store so a crash or a re-run resumes instead of retraining; the
# keys are the paper's configuration names; the volume control is
# 'Unguarded' (a uniform sample of the gate's size)
CELLS_PATH = Path("results/set_f/e6_dopanim_cells.json")
FEAT_DIR = Path("results/set_e/_features")
# Quarter volume is dropped deliberately: the selective set is held to three
# methods (the calibrated gate, the top-decile cut, the top-half cut), each
# with its volume-matched random baseline. Keep this list and the testbeds'
# cell sets consistent -- adding a volume here re-fits every seed.
DEFAULT_COVERAGES = (0.1, 0.5)
# The declared F protocol admits at >=95% precision on a held-out split. On
# dopanim's instance-dependent noise that target can be unreachable at every
# threshold, in which case the four configurations degenerate (nothing is
# admitted and gated == random == unguarded == V0 by construction). The curve
# below is reported either way and the study falls back to the highest target
# that *is* reachable, so the destructive regime still gets a comparison.
IMG_SIZE = 224
LATENT_DIM = 64
SPLIT_SEED = 42          # index-fixed across corrector seeds (E dopanim recipe)


def _nhwc(x: np.ndarray) -> np.ndarray:
    """(N,3,H,W) -> (N,H,W,C): the layout the trainer and the protocol expect.

    ``load_dopanim_view`` returns CIFAR-normalised NCHW; the LSNPC trainer and
    ``compute_stream_scores`` both permute (0,3,1,2) internally on the
    assumption of NHWC input, so the conversion has to happen here.
    """
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        return x.transpose(0, 2, 3, 1)
    return x


def _encode_pixel_latents(vae, X_nhwc: np.ndarray, device: str, bs: int = 256
                    ) -> np.ndarray:
    """Deterministic z_vae latents (conv-VAE mu) for NHWC pixel inputs."""
    return _encode_vae_latents(vae, np.ascontiguousarray(X_nhwc), device,
                               batch_size=bs).cpu().numpy()


def _cached_pixel_vae(X_nhwc: np.ndarray, device: str):
    """The cached conv pixel VAE of the 224px dopanim fit (same key as the
    emission and the Set E runner)."""

    config = SimpleNamespace(vae_ckpt=None, batch_size=64, hidden_dim=256,
                             vae_epochs=30, latent_dim=LATENT_DIM)
    key = dict(dataset="dopanim", noise_type="symmetric", noise_rate=0.0,
               latent_dim=LATENT_DIM,
               img_size=IMG_SIZE)
    path = ckpt_path(**key, ckpt_dir=Path("results/ckpt"))
    if not Path(path).exists():
        raise FileNotFoundError(
            f"needs the cached 224px dopanim pixel VAE {path} — run a dopanim "
            "image_lsnpc cell (e.g. bash shells/set_c_run.sh cc) first.")
    return get_or_train_vae(
        dataset="dopanim", noise_type="symmetric", noise_rate=0.0,
        X_train=np.ascontiguousarray(X_nhwc), config=config,
        device=device, latent_dim=LATENT_DIM,
        )


# The gate axis on dopanim is the SELF-referenced minimality, matching the E
# dopanim recipe (text axis = minimality_self). The oracle-referenced variant
# needs per-row clean labels and on this testbed the only clean labels on the
# annotated pool are class-directory labels the pre-registered E2 decision
# does not trust; the self variant is deployable and needs none.
POOL_AXIS = "minimality(1-minf)_self"


def _m1(lSNPC, X, y_clean, y_noisy, device, bs, M, seed, tag, z_vae=None):
    """Gate axis + corrected label on a candidate set, in one full pass.

    ``compute_stream_scores`` with ``which=(POOL_AXIS,)`` evaluates the
    self-referenced minimality plus the corrected label, so it costs four
    re-decodes per row instead of the full robustness sweep. ``y_clean`` is
    passed only so the corrected-validity diagnostic can be printed; the score
    itself never consults it on this testbed.
    """
    out = compute_stream_scores(
        lSNPC, X, y_noisy, device, batch_size=bs, M=M, seed=seed,
        which=(MINIMALITY_KEY,), y_clean=y_clean, z_vae=z_vae)
    m1 = np.asarray(out["scores"][POOL_AXIS], dtype=np.float64)
    succ = np.asarray(out["succ"], dtype=bool)
    corr = np.asarray(out["corr"]).astype(np.int64)
    print(f"[E6d] {tag}: n={len(X)} corrected-valid={succ.mean():.3f} "
          f"edits={int(out['edited'].sum())}", flush=True)
    return m1, succ, corr


def run(seeds, epochs, beta, clean_set_size=2000, val_size=2000,
        eval_slice=1000, batch=64, coverages=DEFAULT_COVERAGES, downstream=1,
        only_configs=None, cal_size=1500):
    results = {c: [] for c in CONFIG_NAMES}
    cells = {}
    if CELLS_PATH.exists():
        try:
            cells = json.loads(CELLS_PATH.read_text())
        except Exception:
            cells = {}
    for seed in seeds:
        X_pool, y_noisy_pool, y_clean_pool, X_test, y_test = \
            e_common.load_dopanim_view(IMG_SIZE)
        X_pool, X_test = _nhwc(X_pool), _nhwc(X_test)   # NCHW -> NHWC
        accel0 = Accelerator()
        device0 = str(accel0.device)
        # Pixel mode: the protocol needs the VAE latent latents as the
        # correction input; without them it flattens the image (224*224*3) and
        # the correction encoder rejects the shape.
        vae = _cached_pixel_vae(X_pool, device0)
        Z_pool = _encode_pixel_latents(vae, X_pool, device0, batch)
        Z_test = _encode_pixel_latents(vae, X_test, device0, batch)
        cs_idx, test_idx = e_common.dopanim_test_carve(
            seed, eval_slice=eval_slice, val_size=val_size,
            clean_set=clean_set_size, split_seed=SPLIT_SEED,
            allow_large_eval=False)
        ev_idx = test_idx[:eval_slice]
        Xe, yce = X_test[ev_idx], y_test[ev_idx]
        Xcs, ycs = X_test[cs_idx], y_test[cs_idx]
        # Gate calibration source: the guaranteed-clean test rows the carve
        # leaves unused (test_idx minus the eval slice). Out-of-sample for
        # round-0, disjoint from evaluation, and the only labels on this
        # testbed the E2 decision trusts -- the pool's own class-directory
        # labels are never used to calibrate the gate here.
        del X_test                     # ~2.7 GB at 224px; slices are copies

        # Model selection needs a REGIME-MATCHED validation split: held-out pool
        # rows, noisy labels and all, exactly as the Set E recipe carves them.
        # Using the clean test carve as the validation split instead (the first
        # attempt at this harness) selects the checkpoint that never changes a
        # label, because every validation input there is already correct -- it
        # edited 191 of 10,484 rows and scored val_err_h 0.008 against 0.184 in
        # E, i.e. the corrector degenerated to the identity map. Validation and
        # gate calibration are separate concerns: the gate is still calibrated
        # on the guaranteed-clean carve, the fits are selected on the pool.
        # Views, never copies: at 224px a boolean-mask copy costs ~5 GB and
        # stacking such copies is what OOM-killed the first attempts here.
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(X_pool))
        X_pool = np.ascontiguousarray(X_pool[perm])
        y_clean_pool, y_noisy_pool = y_clean_pool[perm], y_noisy_pool[perm]
        Z_pool = Z_pool[perm]
        # Two disjoint pool holdouts, then the training rest. The fit validation
        # selects checkpoints; gate calibration is separate, because a threshold
        # tuned on the rows that selected the model is tuned on its own
        # selection. Both holdouts are NOISY pool rows: calibrating on the
        # already-correct clean carve is degenerate here -- there is nothing to
        # correct there, so every row clears the threshold and coverage
        # saturates at 1.000 with precision 0.987 (measured, twice), which
        # admits the entire pool and empties the training stream. The
        # self-referenced axis is the only one available on this testbed because
        # the pool's own clean labels are class-directory labels E2 distrusts;
        # that caveat is recorded per cell.
        n_tr = len(X_pool) - val_size - cal_size
        X_val, y_val_clean, y_val_noisy = (X_pool[:val_size],
                                          y_clean_pool[:val_size],
                                          y_noisy_pool[:val_size])
        Z_val = Z_pool[:val_size]
        X_calp, y_calp_clean, y_calp_noisy = (
            X_pool[val_size:val_size + cal_size],
            y_clean_pool[val_size:val_size + cal_size],
            y_noisy_pool[val_size:val_size + cal_size])
        Z_calp = Z_pool[val_size:val_size + cal_size]
        Xtr, ytr_clean, ytr_noisy = (X_pool[val_size + cal_size:],
                                     y_clean_pool[val_size + cal_size:],
                                     y_noisy_pool[val_size + cal_size:])
        Ztr = Z_pool[val_size + cal_size:]
        # Downstream features: frozen RN50 at 224px, cached in raw pool order, so
        # the harness's permuted rows map through the same permutation the split
        # used. The eval slice indexes the clean test pool.
        if downstream:
            _fr = np.load(FEAT_DIR / "feat_dopanim_symmetric_rn50_pool_sz224.npy")
            Ftr_feat = _fr[perm][val_size + cal_size:]
            Fe_feat = np.load(
                FEAT_DIR / "feat_dopanim_symmetric_rn50_test_sz224.npy")[ev_idx]
        else:
            Ftr_feat = Fe_feat = None

        n_classes = int(max(y_clean_pool.max(), y_test.max())) + 1
        accel, device = accel0, device0          # constructed once, above
        config = build_config(seed, epochs, beta, batch, clean_set_size,
                              eval_slice, LATENT_DIM, "dopanim", n_classes,
                              img_size=IMG_SIZE, eta=0.1, backbone="swin",
                              pixel=True)
        bs = int(config.batch_size)
        M = int(getattr(config, "iw_samples", 5))

        print(f"\n=== [E6d] dopanim seed={seed} ROUND-0 "
              f"(train={len(Xtr)} val={len(X_val)} cal={len(X_calp)} "
              f"cs={len(cs_idx)}) ===",
              flush=True)
        t0 = e_common.train_round(vae, Xtr, ytr_noisy, Xcs, ycs, ycs, X_val,
                          y_val_clean, y_val_noisy, n_classes, config, accel,
                          device)
        ls0 = t0.model.eval()

        # Full pass over the candidate pool: the M1 gate axis *and* the corrected
        # labels that the admitted rows will carry into round-1.
        m1_pool, succ_pool, corr_pool = _m1(
            ls0, Xtr, ytr_clean, ytr_noisy, device, bs, M, seed,
            "candidate pool", z_vae=Ztr)
        # Calibration on guaranteed-clean test rows: for these rows the noisy
        # label IS the true label, so ``succ`` means 'the corrector left a
        # correct row alone or fixed it', which is exactly the precision the
        # admission threshold is measured at.
        m1_cal, succ_cal, _ = _m1(ls0, X_calp, y_calp_clean, y_calp_noisy,
                                  device, bs, M, seed,
                                  "pool-holdout calibration", z_vae=Z_calp)
        curve, best = e_common.gate_curve(m1_cal, succ_cal)
        print("[E6d] gate curve (held-out split): " + " | ".join(
            (f"prec>={r['target']:.2f}: cov={r['coverage']:.3f} "
             f"prec={r['precision']:.3f}") if r["threshold"] is not None
            else f"prec>={r['target']:.2f}: INFEASIBLE" for r in curve), flush=True)
        print(f"[E6d] best reachable precision={best['precision']:.3f} at "
              f"coverage={best['coverage']:.3f}", flush=True)
        reach = [r for r in curve if r["threshold"] is not None]
        if reach:
            gate_target = reach[0]["target"]          # highest reachable target
            thresh, cov, prec = (reach[0]["threshold"], reach[0]["coverage"],
                                 reach[0]["precision"])
            gated = m1_pool >= thresh
            print(f"[E6d] gate at target={gate_target:.2f}: "
                  f"threshold={thresh:.4f} coverage={cov:.3f} precision={prec:.3f}"
                  f" -> N_gated={int(gated.sum())}", flush=True)
        else:
            gate_target = None
            thresh = cov = prec = None
            gated = np.zeros(len(Xtr), dtype=bool)
            print("[E6d] gate infeasible even at the lowest declared target "
                  "-> no configuration can expand the pool (degenerate by "
                  "construction)", flush=True)
        n_gated = int(gated.sum())

        # round-0 model and the pool latents are no longer needed: at 224px
        # each held copy is GBs and this box OOM-killed the first attempt at
        # ~61 GB resident.
        del ls0, t0, Z_pool
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        _configs = e_common.admission_configurations(m1_pool, gated, n_gated, seed, coverages,
                                   CONFIG_NAMES)
        if only_configs:
            keep_c = {c.strip() for c in only_configs.split(",")}
            _configs = [c for c in _configs if c[0] in keep_c]
        e_common.dump_pool_context("dopanim", seed, corr=corr_pool, m1=m1_pool,
                     succ=succ_pool, y_clean=ytr_clean, y_noisy=ytr_noisy,
                     configs=_configs, perm=perm, val_size=val_size,
                     cal_size=cal_size,
                     eval_idx=ev_idx, cs_idx=cs_idx, test_idx=test_idx)
        for cfg, admit in _configs:
            ckey = f"{seed}:t{gate_target}:{cfg}:d{int(downstream)}"
            if ckey in cells:                  # per-cell idempotent skip
                results.setdefault(cfg, []).append(cells[ckey])
                _vc = cells[ckey].get("valid_clean")
                _txt = f"{_vc:.4f}" if isinstance(_vc, float) else "degenerate"
                print(f"[E6d] {ckey} cached (valid_clean={_txt}) — skip",
                      flush=True)
                continue
            Xa = ya = yan = None
            if cfg == "V0":
                Xc, yc_, ycn = Xcs, ycs, ycs
                desc = "clean set only"
                admit = np.zeros(len(Xtr), dtype=bool)
            else:
                desc = (f"gated N={n_gated}" if cfg == "Gated"
                        else f"{cfg} N={int(admit.sum())}")
                # Downstream use of the corrected labels: an admitted row
                # enters round-1 with the label the corrector emitted for it in
                # the full pass. Passing the ground-truth label here (as the
                # first version did) would turn the study into oracle-label
                # pool expansion and inflate every configuration. Ground truth
                # measurement instrument: the gate's precision and the final
                # accuracy are scored against it, never trained on.
                Xa = Xtr[admit]
                ya = yan = corr_pool[admit]
                Xc = np.concatenate([Xcs, Xa], axis=0)
                yc_ = np.concatenate([ycs, ya], axis=0)
                ycn = np.concatenate([ycs, yan], axis=0)

            # Admitted rows move out of the noisy training pool and into the
            # clean pool: trusting a row's corrected label and still training it
            # against its noisy label is a contradiction.
            keep = ~admit
            if keep.sum() == 0:
                print(f"[E6d] {cfg} admits the whole pool: no noisy rows left "
                      f"to train on -> recorded as degenerate", flush=True)
                cells[ckey] = {"setting": "dopanim", "seed": seed,
                               "config": cfg, "degenerate": True,
                               "n_admitted": int(admit.sum()),
                               "valid_clean": None}
                CELLS_PATH.write_text(json.dumps(cells, indent=2))
                continue
            Xtr_kept, ytr_kept = Xtr[keep], ytr_noisy[keep]
            n_removed = int((~keep).sum())
            print(f"[E6d] seed={seed} config={cfg} pool={desc} "
                  f"clean_set={len(Xc)} train_kept={len(Xtr_kept)}", flush=True)
            cfg2 = build_config(seed, epochs, beta, batch, clean_set_size,
                                eval_slice, LATENT_DIM, "dopanim", n_classes,
                                img_size=IMG_SIZE, eta=0.1, backbone="swin",
                                pixel=True)
            cfg2.output_dir = f"/tmp/e6d_dopanim_s{seed}_b{beta}_e{epochs}_{cfg.lower()}"
            a2 = Accelerator()
            t1 = e_common.train_round(vae, Xtr_kept, ytr_kept, Xc, yc_, ycn, X_val,
                              y_val_clean, y_val_noisy, n_classes, cfg2, a2,
                              str(a2.device))
            out = compute_stream_scores(
                t1.model.eval(), Xe, yce, str(a2.device), batch_size=bs, M=M,
                seed=seed, which=(MINIMALITY_KEY,), y_clean=yce,
                z_vae=Z_test[ev_idx])
            if downstream:
                assert len(Xtr) == len(ytr_noisy), (
                    f"emission inputs/labels misaligned: {len(Xtr)} vs "
                    f"{len(ytr_noisy)}")
                # The 224px pixel path has almost no GPU headroom left after the
                # fit (CUDA OOM at 23.5 GiB with the fit's batch size), so the
                # emission runs at a quarter batch after releasing cached blocks.
                if str(a2.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                r1 = compute_stream_scores(
                    t1.model.eval(), Xtr, ytr_noisy, str(a2.device),
                    batch_size=max(8, bs // 4), M=M, seed=seed,
                    which=(MINIMALITY_KEY,), y_clean=ytr_clean, z_vae=Ztr)
                corr_r1 = np.asarray(r1["corr"]).astype(np.int64)
                del r1
                gc.collect()
                if str(a2.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                # Head on CPU: the 224px pixel path has no GPU headroom left
                # (CUDA OOM at 23.5 GiB), and the head is a small MLP over
                # cached features, so CPU costs a minute and cannot fail.
                ds_res = head_eval(Ftr_feat, corr_r1, Fe_feat, yce, n_classes,
                                   seed, "cpu")
                print(f"[E6d] {cfg} seed={seed}: downstream_acc="
                      f"{ds_res['acc']:.4f} (corrected-pool labels "
                      f"{(corr_r1 == ytr_clean).mean():.3f} valid on the "
                      f"pool's class-dir labels)", flush=True)
            corr = np.asarray(out["corr"])
            n_adm = 0 if cfg == "V0" else int(admit.sum())
            adm_acc = (float((corr_pool[admit] == ytr_clean[admit]).mean())
                       if n_adm else None)
            entry = {"setting": "dopanim", "seed": seed, "config": cfg,
                     "coverage": (float(cfg.split("@")[1]) if "@" in cfg
                           else None),
                     "admitted_precision": (float(succ_pool[admit].mean())
                                            if n_adm else None),
                     "pool_labels": "corrector-full-pass",
                     "gate_axis": POOL_AXIS,
                     "calibration_source": ("held-out noisy pool rows, "
                                             "disjoint from the fit validation "
                                             "split (n=%d); precision measured "
                                             "against the pool's class-dir "
                                             "labels -- E2 caveat"
                                             % len(X_calp)),
                     "admitted_label_acc": adm_acc,
                     "n_admitted": n_adm,
                     "n_train_removed": n_removed,
                     "valid_clean": float((corr == yce).mean()),
                     **({"downstream_acc": ds_res["acc"],
                         "downstream_f1": ds_res["f1"],
                         "r1_pool_label_valid": float((corr_r1 == ytr_clean).mean())}
                        if downstream else {}),
                     "corr_f1": float(multiclass_f1(corr, yce)),
                     "noisy_f1": float(multiclass_f1(yce, yce)),
                     "gate_threshold": thresh, "gate_coverage": cov,
                     "gate_precision": prec, "gate_target": gate_target,
                     "gate_curve": curve,
                     "gate_best_precision": best["precision"],
                     "gate_best_coverage": best["coverage"]}
            results.setdefault(cfg, []).append(entry)
            cells[ckey] = entry
            CELLS_PATH.write_text(json.dumps(cells, indent=2))
            print(f"[E6d] {cfg} seed={seed}: valid_clean="
                  f"{entry['valid_clean']:.4f}", flush=True)
            del t1, Xc, yc_, ycn
            if Xa is not None:
                del Xa, ya, yan
            gc.collect()
            if str(a2.device).startswith("cuda"):
                torch.cuda.empty_cache()

    out_path = Path("results/set_f/e6_dopanim_summary.json")
    prior = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text()).get("configs", {})
        except Exception:
            prior = {}
    summary = {"setting": "dopanim", "noise_type": "real (crowd, 3-10 annotators)",
               "configs": {}}
    arm_names = [c for c in CONFIG_NAMES if results.get(c)]
    arm_names += sorted(c for c in results if c not in CONFIG_NAMES
                        and results[c])
    for cfg in arm_names:
        rows = results[cfg]
        if not rows:
            if cfg in prior:
                summary["configs"][cfg] = prior[cfg]
            continue
        vc = [r["valid_clean"] for r in rows]
        f1 = [r["corr_f1"] for r in rows]
        if cfg in prior:                       # merge earlier seed batches
            pv = prior[cfg]
            vc = list(pv.get("valid_clean", {}).get("per_seed", [])) + vc
            f1 = list(pv.get("corr_f1", {}).get("per_seed", [])) + f1
        summary["configs"][cfg] = {
            "valid_clean": {"mean": float(np.mean(vc)), "std": float(np.std(vc)),
                            "per_seed": vc},
            "corr_f1": {"mean": float(np.mean(f1)), "std": float(np.std(f1)),
                        "per_seed": f1}}
    for a, b, label in (("Gated", "Unguarded", "gated - unguarded (calibrated)"),
                        ("c@0.1", "f@0.1", "c@0.1 - f@0.1"),
                        ("c@0.25", "f@0.25", "c@0.25 - f@0.25"),
                        ("c@0.5", "f@0.5", "c@0.5 - f@0.5")):
        av = summary["configs"].get(a, {}).get("valid_clean", {}).get("mean")
        bv = summary["configs"].get(b, {}).get("valid_clean", {}).get("mean")
        if av is not None and bv is not None:
            summary[f"{a}_minus_{b}"] = float(av - bv)
            print(f"=== [E6d] {label} = {av - bv:+.4f} ===", flush=True)
    g = summary["configs"].get("Gated", {}).get("valid_clean", {})
    b = summary["configs"].get("Unguarded", {}).get("valid_clean", {})
    if g and b:
        delta = float(np.mean(g["per_seed"]) - np.mean(b["per_seed"]))
        summary["gated_minus_unguarded_valid_clean"] = delta
        print(f"\n=== [E6d] dopanim gated - unguarded (valid_clean) = {delta:+.4f} ===")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--clean-set-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--eval-slice", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--downstream", type=int, default=1,
                    help="train the downstream head on the corrected dataset")
    ap.add_argument("--only-configs", default=None,
                    help="comma list: run only these configurations")
    ap.add_argument("--coverages", nargs="+", type=float, default=list(DEFAULT_COVERAGES),
                    help="admission fractions for the selectivity sweep")
    args = ap.parse_args()
    run(args.seeds, args.epochs, args.beta, args.clean_set_size,
        args.val_size, args.eval_slice, args.batch, args.coverages, args.downstream,
        args.only_configs)


if __name__ == "__main__":
    main()
