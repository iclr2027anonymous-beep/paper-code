#!/usr/bin/env python3
"""Set E — emission mode (ENG-2): per-row corrections + five scores on a pool.

Loads a trained LSNPC corrector bundle (results/ckpt/lsnpc/...) and emits, for
every row of the requested pool, the row's stream label, corrected label and
the per-query scores of the experiment protocol §0 — computed from the
correction path alone (methodology §4: "never from a clean label"):

    confidence p* (p_max(conf)), self-consistent minimality
    1-f*_self (partial-edit re-runs reproducing the row's own corrected
    label), self-consistent noise-robustness r_self (K=32 perturbations
    re-decoding to the corrected label), plausibility pi.

When the pool rows carry a clean oracle (NoisyAG, CIFAR-N), the clean-
referenced protocol scores + transitions (mis/succ, WR/RW/RR/WW) are ALSO
stored under diagnostic keys — they feed the held-out label-level sign check
and transfer audits only, never the gate.

Rows are emitted in groups (eval, val, cs, tr — bundle split indices; cs is
empty for dopanim, whose clean set lives in the clean test pool). The
retrain pool (configs a-f) = cs + tr rows (is_pool mask); val + eval rows =
is_holdout, the honest out-of-sample label-level sign check. EXCEPTION:
dopanim (E2) retrains on the FULL annotated pool, so every annotated row is
is_pool there; is_holdout (eval+val) is still marked for the out-of-sample
sign check of the corrector's own holdout rows.

Usage (house convention: launch through the HF launcher, never bare python):
    accelerate launch --mixed_precision=bf16 -m scripts.emit_lsnpc_stream \
        --ckpt results/ckpt/lsnpc/text/lsnpc_noisyag_worst_s42_n0.0_symmetric_e15_b0.5_cs2000.pt \
        --outdir results/set_e/e_text_noisyag_worst_cs42 \
        [--modality text|image] [--batch-size 256] [--iw-m 5] [--rob-k 32]

Outputs: emissions_cs<corrseed>.npz (row-aligned arrays) + .json (meta).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.protocol import compute_stream_scores
from scripts import e_common
from scripts.lsnpc_ckpt import load_bundle
from trainers.lsnpc import _encode_vae_latents

MODALITIES = ("text", "image")


def _resolve_device(dev: str | None) -> str:
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _nhwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        return x.transpose(0, 2, 3, 1)
    return x


def _encode_pixel_latents(vae, X_nhwc: np.ndarray, device: str,
                    bs: int = 256) -> np.ndarray:
    """Deterministic z_vae latents (conv-VAE mu) for NHWC pixel inputs."""
    return _encode_vae_latents(vae, np.ascontiguousarray(X_nhwc), device,
                               batch_size=bs).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--modality", choices=MODALITIES, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--iw-m", type=int, default=5)
    ap.add_argument("--rob-k", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--corr-seed", type=int, default=42)
    ap.add_argument("--robustness-seed", type=int, default=None,
                    help="RNG seed for the K perturbation draws; defaults to "
                         "the corrector run's seed so the eval-group "
                         "robustness matches the archived per-query npz.")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.rob_k < 1:
        raise ValueError(f"--rob-k must be >= 1 (got {args.rob_k})")
    device = _resolve_device(args.device)
    bundle = load_bundle(args.ckpt, device)
    model = bundle["model"]
    splits = bundle["splits"]
    run = bundle["run"]
    arch = bundle["arch"]["build_kwargs"]
    modality = args.modality or ("image" if arch.get("image_data")
                                 or run.get("dataset") in e_common.IMAGE_TESTBEDS
                                 else "text")
    dataset = run["dataset"]
    corr_seed = int(run.get("seed", args.corr_seed))
    rob_seed = args.robustness_seed if args.robustness_seed is not None \
        else corr_seed
    n_classes = int(arch["n_classes"])

    print(f"[emit] {args.ckpt.name}: modality={modality} dataset={dataset} "
          f"corr_seed={corr_seed} rob_seed={rob_seed} arch_image_data="
          f"{arch.get('image_data')} latent={arch.get('latent_dim')}",
          flush=True)

    # ── data view ───────────────────────────────────────────────────
    pixel = bool(arch.get("image_data"))
    if modality == "text":
        X_all, yc_all, yn_all = e_common.load_text_view(dataset)
        z_none = True
    elif dataset in ("cifar10n", "cifar100n"):
        view = run.get("noise_type") or run.get("cell_tag") or "worse"
        if view not in ("worse", "fine", "aggre", "coarse"):
            view = "worse" if dataset == "cifar10n" else "fine"
        if dataset == "cifar10n":
            view = "worse" if view == "worse" else "aggre"
        else:
            view = "fine" if view == "fine" else "fine"
        X_all, yn_all, yc_all, X_test, y_test = e_common.load_cifar_view(
            dataset, view)
        if not pixel:
            F_all = e_common.rn50_features_cached(
                X_all, dataset, view, "pool", device, args.batch_size)
    elif dataset == "dopanim":
        if not pixel:
            raise SystemExit(
                "dopanim E correctors are pixel Swin (224px, C-c). Refusing "
                "to emit an embedding-mode bundle for dopanim.")
        X_all, yn_all, yc_all, X_test, y_test = e_common.load_dopanim_view()
    elif dataset == "animal10n":
        # Same E2 shape as dopanim: annotations live on the train split only,
        # so the clean set comes from the clean test pool and yc_all is None.
        # Unlike dopanim the correctors here are frozen-RN50 embedding mode,
        # so the embedding path is the supported one.
        X_all, yn_all, yc_all, X_test, y_test = e_common.load_animal10n_view()
        if not pixel:
            F_all = e_common.rn50_features_cached(
                X_all, dataset, "native", "pool", device, args.batch_size)
    else:
        raise SystemExit(f"unknown dataset {dataset}")

    # ── groups from the bundle splits (dopanim cs lives in the test pool) ──
    n = len(X_all)
    groups = []
    for name, key in (("eval", "eval_idx"), ("val", "val_idx"),
                      ("cs", "cs_idx"), ("tr", "tr_idx")):
        idx = splits.get(key)
        if idx is None or len(idx) == 0:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        if name == "cs" and dataset in ("dopanim", "animal10n"):
            continue  # cs rows are clean test-pool rows, not annotated pool
        if int(idx.max()) >= n:
            continue  # indices outside the annotated pool (dopanim cs)
        groups.append((name, idx))
    if not groups:
        raise SystemExit("no usable split indices in bundle")

    # ── pixel latents (once over the pool) ───────────────────────────
    vae = None
    zv_cache = {}
    if pixel:
        size = int(arch.get("img_size", X_all.shape[-1]))
        if X_all.shape[-1] != size:
            raise SystemExit(
                f"bundle img_size {size} != data size {X_all.shape[-1]}; "
                "re-prep the data at the bundle resolution "
                "(scripts/dopanim_prep.py --size).")
        # pixel VAE trained over the annotated pool at fit time
        vae = e_common.pixel_vae(bundle, _nhwc(X_all), device,
                          int(arch.get("latent_dim", 64)), size)

    rows_list = []
    meta = {}
    for name, idx in groups:
        if pixel:
            Xg = _nhwc(np.ascontiguousarray(X_all[idx]))
            zv = zv_cache.get(name)
            if zv is None:
                zv = _encode_pixel_latents(vae, Xg, device, args.batch_size)
                zv_cache[name] = zv
            x_in = Xg
            z_in = zv
        elif modality == "text":
            x_in = np.ascontiguousarray(X_all[idx])
            z_in = None
        else:  # image embedding mode: features are the latent
            x_in = np.ascontiguousarray(F_all[idx])
            z_in = None
        yn = np.asarray(yn_all[idx]).astype(np.int64)
        yc = (np.asarray(yc_all[idx]).astype(np.int64)
              if yc_all is not None else None)
        out = compute_stream_scores(
            model, x_in, yn, device, batch_size=args.batch_size,
            M=args.iw_m, seed=rob_seed, k_perturb=args.rob_k, z_vae=z_in,
            y_clean=yc)
        rows_list.append((name, idx, out))
        trans = out.get("transitions") or {}
        meta[name] = {
            "n": int(len(idx)),
            "edited": int(out["edited"].sum()),
            "corr_acc": (float((out["corr"] == yc).mean())
                         if yc is not None else None),
            "mis_frac": (float((yn != yc).mean())
                         if yc is not None else None),
            "transitions": trans,
        }
        print(f"[emit] group {name}: n={len(idx)} edited={out['edited'].sum()} "
              f"transitions={trans} mis_frac={meta[name]['mis_frac']}",
              flush=True)

    # ── concatenate row-aligned arrays ───────────────────────────────
    names = [g[0] for g in rows_list]
    order = np.concatenate([g[1] for g in rows_list]).astype(np.int64)
    group_codes = np.concatenate([np.full(len(g[1]), i, dtype=np.int64)
                                  for i, g in enumerate(rows_list)])
    group_name = {i: nm for i, nm in enumerate(names)}
    n_tot = len(order)
    yn_all_rows = np.concatenate(
        [np.asarray(yn_all[g[1]]) for g in rows_list]).astype(np.int64)
    y_clean_rows = (np.concatenate(
        [np.asarray(yc_all[g[1]]) for g in rows_list]).astype(np.int64)
        if yc_all is not None else None)

    def _col(key):
        """Top-level per-row array from the per-group compute_stream_scores
        outputs (also reads inside the nested 'scores' dict)."""
        parts = []
        for _, _, o in rows_list:
            if key in o:
                parts.append(np.asarray(o[key]))
            elif (o.get("scores") or {}).get(key) is not None:
                parts.append(np.asarray(o["scores"][key]))
            else:
                raise KeyError(f"{key} missing from a group emission")
        return np.concatenate(parts)

    payload = {
        "row_global": order,
        "group": group_codes,
        "group_name": json.dumps(group_name),
        "y_noisy": yn_all_rows,
        "corr": _col("corr").astype(np.int64),
        "edited": _col("edited").astype(bool),
        "is_pool": np.zeros(n_tot, dtype=bool),
        "is_holdout": np.zeros(n_tot, dtype=bool),
    }
    for nm in ("cs", "tr"):
        if nm in names:
            payload["is_pool"] |= np.isin(group_codes, [names.index(nm)])
    for nm in ("eval", "val"):
        if nm in names:
            payload["is_holdout"] |= np.isin(group_codes, [names.index(nm)])
    if dataset in ("dopanim", "animal10n"):
        # E2 carve (dopanim, animal10n): the retrain stream is the FULL
        # annotated pool
        # (10,484 rows) — the corrector's eval/val carves are only for its
        # own fit/checkpoint selection; every annotated row is emitted and
        # downstream configs a-f train on all of them. is_holdout keeps the
        # corrector's eval+val rows for the out-of-sample label-level sign
        # check (those corrected labels are out-of-sample w.r.t. the
        # corrector even though they join the downstream pool).
        pool_groups = [names.index(nm) for nm in ("eval", "val", "tr")
                       if nm in names]
        if pool_groups:
            payload["is_pool"] |= np.isin(
                group_codes, pool_groups)
    if y_clean_rows is not None:
        payload["y_clean"] = y_clean_rows
        payload["mis"] = yn_all_rows != y_clean_rows
        payload["succ"] = payload["corr"] == y_clean_rows
    # Score axes (nested under scores in the compute output).
    for k in sorted({k for _, _, o in rows_list for k in (o.get("scores") or {})}):
        payload[f"score__{k}"] = _col(k)
    np.savez_compressed(outdir / f"emissions_cs{corr_seed}.npz", **payload)

    info = {
        "ckpt": str(args.ckpt), "modality": modality, "dataset": dataset,
        "corr_seed": corr_seed, "robustness_seed": rob_seed,
        "n_classes": n_classes, "pixel": pixel, "img_size":
            int(arch.get("img_size", 0)) if pixel else None,
        "n_rows": int(n_tot), "groups": meta,
        "group_order": names,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()
                        if v is not None},
        "arch": arch,
        "run": {k: v for k, v in run.items() if isinstance(v, (str, int, float))},
    }
    (outdir / f"emissions_cs{corr_seed}.json").write_text(
        json.dumps(info, indent=2, default=str))
    print(f"[emit] wrote {outdir / ('emissions_cs%d.npz' % corr_seed)} "
          f"(n={n_tot})")


if __name__ == "__main__":
    main()
