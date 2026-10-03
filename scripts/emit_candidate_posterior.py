#!/usr/bin/env python3
"""Set K — candidate-posterior emission (the manuscript appendix C, app:ceiling).

Appendix C ("The Repair Ceiling of a Single-Proposal Rule and an Open
Extension") states that a rule whose only actions are to keep the stream label
``l_i`` or adopt the single proposal ``c_i = C(x_i, l_i)`` can never repair a row
the corrector misses: on ``S \\ S_1`` both ``l_i`` and ``c_i`` are wrong.  The
open extension widens the candidate set to the K most probable classes of the
corrected-label posterior

    p_c(y | x, l) ~= sum_k w_k [softmax(g_phi(x, z_k))]_y ,        (eq:post_marg)

the importance-weighted (IW) average of per-sample softmaxes at the corrected
latent samples ``z_k`` with sample weights ``w_k``.  The appendix is explicit
that averaging at the *label* level is required: evaluating the expectation at
the mean corrected latent is a biased plug-in estimate because the softmax is
nonlinear.  This script emits BOTH estimators per row so the bias is measurable.

Per row it writes (row-aligned, one npz per cell):

    y_noisy      stream label l_i
    y_clean      oracle label (diagnostics only; never used by the rule)
    top_idx      (n, K) classes of p_c in descending posterior order
    top_p        (n, K) their IW-averaged probabilities
    corr_iw      argmax of the IW posterior  (the single proposal c_i)
    corr_plugin  argmax of the plug-in posterior (mean-latent estimate)
    p_cur_probe  independent classifier's probability of the CURRENT label
                 (guard signal: "an independent classifier is already
                 confident in the current label" withholds a relabel)
    edited       corr_iw != y_noisy
    group        split group code (eval/val/cs/tr)
    is_pool      cs+tr rows (the retrain stream)
    row_global   index into the annotated pool

The independent classifier is a multinomial logistic regression trained on the
SAME frozen features with the SAME noisy labels (oracle-free, and independent of
the correction path).  It is unavailable in pixel mode (the corrector input is an
image, not a feature vector); those cells record ``probe = "unavailable"`` and
the analyzer evaluates guard-free rules only.

Determinism: the correction path samples z_k from the global RNG, so the script
seeds torch/numpy with the corrector seed before inference.  Same seed + same
bundle => same emission.

Usage (house convention: launch through the HF launcher, never bare python):
    accelerate launch --mixed_precision=bf16 -m scripts.emit_candidate_posterior \\
        --ckpt results/ckpt/lsnpc/text/lsnpc_noisyag_worst_s45_n0.0_symmetric_e15_b0.5_cs2000.pt \\
        --outdir results/set_k/k_text_noisyag_worst_cs45 \\
        [--modality text|image] [--topk 10] [--iw-m 5] [--batch-size 256]

Outputs: candidates_cs<corrseed>.npz + candidates_cs<corrseed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import e_common
from scripts.lsnpc_ckpt import load_bundle
from trainers.lsnpc import _encode_vae_latents
from utils.batching import place_loader_local, tensor_loader

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
    return _encode_vae_latents(vae, np.ascontiguousarray(X_nhwc), device,
                               batch_size=bs).cpu().numpy()


@torch.no_grad()
def _iw_posterior(model, X: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor,
                  device: str, batch_size: int, M: int) -> dict:
    """IW-averaged class posterior (eq:post_marg) + the plug-in estimate.

    Returns per-row arrays: probs_iw (n, C), probs_plugin (n, C).
    """
    n = X.size(0)
    iw_parts, pl_parts = [], []
    dl = place_loader_local(tensor_loader(X, Z, Y, batch_size=batch_size,
                                          shuffle=False), device)
    for xb, zb, yb in dl:
        log_w, z_samples = model.iw_log_weights(xb, zb, yb, M)   # (B,K),(K,B,D)
        w = torch.softmax(log_w, dim=-1)                          # (B,K)
        acc = None
        for k in range(M):
            pk = model.corrected_conditioning(xb, z_samples[k]).float()  # (B,C)
            acc = pk * w[:, k:k + 1] if acc is None else acc + pk * w[:, k:k + 1]
        iw_parts.append(acc.detach().float().cpu().numpy())
        z_mean = model.importance_weighted_mean(log_w, z_samples)
        pl = model.corrected_conditioning(xb, z_mean).detach().float().cpu().numpy()
        pl_parts.append(pl)
    return {"probs_iw": np.concatenate(iw_parts, axis=0),
            "probs_plugin": np.concatenate(pl_parts, axis=0)}


def _train_probe(F: np.ndarray, y_noisy: np.ndarray, train_mask: np.ndarray,
                 max_iter: int = 200):
    """Independent classifier: multinomial logistic regression on frozen
    features with the NOISY labels (oracle-free, independent of the corrector).

    Returns ``(proba, info)`` where ``proba(X, cls_idx)`` gives (n, K) class
    probabilities for the requested class indices.  The guard compares the
    probe's probability of the CURRENT label against its probability of a
    CANDIDATE, a scale-free test: a plain logistic head is uncalibrated, so an
    absolute threshold on it degenerates differently per class count (measured:
    96.8% of CIFAR-100N's wrong rows sit above 0.9, 0.3% for CIFAR-10N).
    """

    Xtr = np.asarray(F[train_mask], dtype=np.float64)
    ytr = np.asarray(y_noisy[train_mask]).astype(np.int64)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(scaler.transform(Xtr), ytr)
    info = {"n_train": int(train_mask.sum()),
            "train_acc": float((clf.predict(scaler.transform(Xtr)) == ytr).mean()),
            "n_classes": int(clf.classes_.size), "max_iter": max_iter}
    col = {int(c): j for j, c in enumerate(clf.classes_)}

    def proba(X: np.ndarray, cls_idx: np.ndarray) -> np.ndarray:
        """(n, K) probe probabilities of the requested class indices."""
        P = clf.predict_proba(scaler.transform(np.asarray(X, dtype=np.float64)))
        cls = np.asarray(cls_idx, dtype=np.int64)
        out = np.full(cls.shape, np.nan)
        for j in range(cls.shape[1]):
            cols = [col.get(int(c), -1) for c in cls[:, j]]
            for i, cc in enumerate(cols):
                if cc >= 0:
                    out[i, j] = P[i, cc]
        return out

    return proba, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--modality", choices=MODALITIES, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--iw-m", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--corr-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the correction draws; defaults to the "
                         "corrector run's seed (same convention as the E emit).")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the independent-classifier guard signal")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    bundle = load_bundle(args.ckpt, device)
    model, splits, run = bundle["model"], bundle["splits"], bundle["run"]
    arch = bundle["arch"]["build_kwargs"]
    modality = args.modality or ("image" if arch.get("image_data")
                                 or run.get("dataset") in e_common.IMAGE_TESTBEDS
                                 else "text")
    dataset = run["dataset"]
    corr_seed = int(run.get("seed", args.corr_seed))
    seed = args.seed if args.seed is not None else corr_seed
    n_classes = int(arch["n_classes"])
    K = int(args.topk)

    print(f"[cand] {args.ckpt.name}: modality={modality} dataset={dataset} "
          f"corr_seed={corr_seed} seed={seed} K={K} M={args.iw_m} "
          f"latent={arch.get('latent_dim')}", flush=True)

    # ── data view (mirrors the Set E emission) ─────────────────────────
    pixel = bool(arch.get("image_data"))
    if modality == "text":
        X_all, yc_all, yn_all = e_common.load_text_view(dataset)
        F_all = None
    elif dataset in ("cifar10n", "cifar100n"):
        view = run.get("noise_type") or run.get("cell_tag") or "worse"
        if dataset == "cifar10n":
            view = "worse" if view == "worse" else "aggre"
        else:
            view = "fine"
        X_all, yn_all, yc_all, _X_test, _y_test = e_common.load_cifar_view(
            dataset, view)
        F_all = None
        if not pixel:
            F_all = e_common.rn50_features_cached(
                X_all, dataset, view, "pool", device, args.batch_size)
    elif dataset == "dopanim":
        if not pixel:
            raise SystemExit(
                "dopanim correctors are pixel Swin — refusing an embedding-"
                "mode bundle for dopanim.")
        X_all, yn_all, yc_all, _X_test, _y_test = e_common.load_dopanim_view()
        F_all = None
    else:
        raise SystemExit(f"unknown dataset {dataset}")

    n = len(X_all)
    groups = []
    for name, key in (("eval", "eval_idx"), ("val", "val_idx"),
                      ("cs", "cs_idx"), ("tr", "tr_idx")):
        idx = splits.get(key)
        if idx is None or len(idx) == 0:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        if name == "cs" and dataset == "dopanim":
            continue
        if int(idx.max()) >= n:
            continue
        groups.append((name, idx))
    if not groups:
        raise SystemExit("no usable split indices in bundle")

    vae = None
    zv_cache: dict[str, np.ndarray] = {}
    if pixel:
        size = int(arch.get("img_size", X_all.shape[-1]))
        if X_all.shape[-1] != size:
            raise SystemExit(
                f"bundle img_size {size} != data size {X_all.shape[-1]}; "
                "re-prep the data at the bundle resolution.")
        vae = e_common.pixel_vae(bundle, _nhwc(X_all), device,
                          int(arch.get("latent_dim", 64)), size)

    # ── independent classifier (guard signal) ──────────────────────────
    # Trained on the correction pool (cs+tr) features with their NOISY labels.
    yc_np = np.asarray(yc_all).astype(np.int64) if yc_all is not None else None
    pool_mask = np.zeros(n, dtype=bool)
    for nm in ("cs", "tr"):
        for gname, gidx in groups:
            if gname == nm:
                pool_mask[gidx] = True
    probe_fn, probe_info = None, {"probe": "unavailable"}
    if not args.no_probe:
        feats = X_all if F_all is None else F_all
        if feats.ndim != 2:
            probe_info = {"probe": "unavailable",
                          "reason": "pixel-mode input is not a feature vector"}
        else:
            try:
                probe_fn, pinfo = _train_probe(feats, np.asarray(yn_all), pool_mask)
                probe_info = {"probe": "logreg_on_frozen_features", **pinfo}
            except Exception as exc:            # noqa: BLE001 — recorded, not hidden
                probe_info = {"probe": "failed", "error": str(exc)}
        feats = None
    print(f"[cand] probe: {probe_info}", flush=True)

    # ── per-group emission ─────────────────────────────────────────────
    parts: dict[str, list] = {}
    meta: dict[str, dict] = {}

    def _push(key: str, arr) -> None:
        parts.setdefault(key, []).append(np.asarray(arr))

    torch.manual_seed(seed)
    np.random.seed(seed)
    for name, idx in groups:
        if pixel:
            Xg = _nhwc(np.ascontiguousarray(X_all[idx]))
            zv = zv_cache.get(name)
            if zv is None:
                zv = _encode_pixel_latents(vae, Xg, device, args.batch_size)
                zv_cache[name] = zv
            x_in = np.ascontiguousarray(Xg)
            z_in = zv
        elif modality == "text":
            x_in = np.ascontiguousarray(X_all[idx])
            z_in = None
        else:
            x_in = np.ascontiguousarray(F_all[idx])
            z_in = None
        yn = np.asarray(yn_all[idx]).astype(np.int64)
        yc = yc_np[idx] if yc_np is not None else None
        Xt = torch.as_tensor(x_in, dtype=torch.float32, device=device)
        Zt = torch.as_tensor(
            z_in if z_in is not None else x_in.reshape(len(x_in), -1),
            dtype=torch.float32, device=device)
        Yt = torch.as_tensor(yn, dtype=torch.long, device=device)

        out = _iw_posterior(model, Xt, Zt, Yt, device, args.batch_size,
                            int(args.iw_m))
        probs = out["probs_iw"]
        order = np.argsort(-probs, axis=1)[:, :K]
        top_p = np.take_along_axis(probs, order, axis=1)
        corr_iw = order[:, 0].astype(np.int64)
        corr_pl = out["probs_plugin"].argmax(-1).astype(np.int64)
        p_cur_post = probs[np.arange(len(yn)), yn]      # p_c(current label)

        _push("top_idx", order.astype(np.int16))
        _push("top_p", top_p.astype(np.float32))
        _push("p_cur_post", p_cur_post.astype(np.float32))
        _push("corr_iw", corr_iw)
        _push("corr_plugin", corr_pl)
        _push("y_noisy", yn)
        if yc is not None:
            _push("y_clean", yc)
        if probe_fn is not None:
            feats_g = (X_all[idx] if F_all is None else F_all[idx])
            _push("p_cur_probe",
                  probe_fn(feats_g, yn[:, None])[:, 0].astype(np.float32))
            _push("probe_top_p",
                  probe_fn(feats_g, order).astype(np.float32))
        _push("edited", (corr_iw != yn))
        _push("row_global", idx)

        trans = None
        if yc is not None:
            trans = {"RR": int(((yn == yc) & (corr_iw == yc)).sum()),
                     "RW": int(((yn == yc) & (corr_iw != yc)).sum()),
                     "WR": int(((yn != yc) & (corr_iw == yc)).sum()),
                     "WW": int(((yn != yc) & (corr_iw != yc)).sum())}
        meta[name] = {"n": int(len(idx)), "edited": int((corr_iw != yn).sum()),
                      "transitions": trans,
                      "plugin_vs_iw_disagree": int((corr_iw != corr_pl).sum())}
        print(f"[cand] group {name}: n={len(idx)} edited={meta[name]['edited']} "
              f"transitions={trans} iw!=plugin={meta[name]['plugin_vs_iw_disagree']}",
              flush=True)

    group_codes = np.concatenate([np.full(len(g[1]), i, dtype=np.int64)
                                  for i, g in enumerate(groups)])
    names = [g[0] for g in groups]
    payload = {k: np.concatenate(v) for k, v in parts.items()}
    payload["group"] = group_codes
    payload["group_order"] = np.array(json.dumps(names))
    payload["is_pool"] = np.isin(group_codes,
                                 [names.index(nm) for nm in ("cs", "tr")
                                  if nm in names])
    payload["is_holdout"] = np.isin(group_codes,
                                    [names.index(nm) for nm in ("eval", "val")
                                     if nm in names])
    np.savez_compressed(outdir / f"candidates_cs{corr_seed}.npz", **payload)

    info = {"ckpt": str(args.ckpt), "modality": modality, "dataset": dataset,
            "corr_seed": corr_seed, "seed": seed, "topk": K, "iw_m": int(args.iw_m),
            "n_classes": n_classes, "pixel": pixel,
            "n_rows": int(sum(len(g[1]) for g in groups)),
            "groups": meta, "group_order": names, "probe": probe_info,
            "arch": arch,
            "run": {k: v for k, v in run.items() if isinstance(v, (str, int, float))}}
    (outdir / f"candidates_cs{corr_seed}.json").write_text(
        json.dumps(info, indent=2, default=str))
    print(f"[cand] wrote {outdir / ('candidates_cs%d.npz' % corr_seed)} "
          f"(n={len(payload['y_noisy'])})", flush=True)


if __name__ == "__main__":
    main()
