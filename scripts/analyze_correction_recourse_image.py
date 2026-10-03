#!/usr/bin/env python3
"""Pure-LSNPC correction-as-recourse evaluation for IMAGE data (CIFAR-10/10N).

Image counterpart of the tabular analyzer (analyze_correction_recourse.py):
same E1-E5 experiments, same rec-cache schema, so the gate scripts
(evaluate_gate_v3 / s2a) load it unchanged. The expensive shared step
(classifiers + Conv VAE + LSNPC stage-1 + correction path) runs once per
cell and is cached to rec_cache_<cell>_nt<n_test>.json.

Usage:
    python -m scripts.analyze_correction_recourse_image --experiment E1 \
        --dataset cifar10n --noise 0.4 --noise-type worse \
        --seed 42 --use-semi --lsnpc-beta 0.5 --vae-epochs 200 \
        --lsnpc-epochs 30 --n-test -1 --output-dir results/recourse

Data layout: NHWC numpy throughout — the CIFAR loader emits NCHW, so
transposed on load. The VAE encoder requires NHWC. ``lsnpc.decode`` returns
the flat ``(B, D)`` decoder mean, which ``clf.predict`` accepts as a
flattened image batch.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from data_process.base import get_dataset
from experiments.arguments import ExperimentConfig
from experiments.data import _select_device
from experiments.lsnpc_stage1 import run_lsnpc_stage1
from trainers.conv_classifier import ConvClassifier, train_conv_classifier
from trainers.plausibility_vae import get_or_train
from utils.batching import place_loader_local, tensor_loader
from utils.conditioning import build_predictive_conditioning

log = logging.getLogger("correction_recourse_image")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "ALL")


def _cell_key(config) -> str:
    return f"{config.dataset}_{config.noise_type}{int(config.noise * 100)}_seed{config.seed}"


def _to_nhwc(x: np.ndarray) -> np.ndarray:
    """NCHW (N,3,H,W) → NHWC (N,H,W,3) if the tensor is 4D NCHW."""
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        return np.transpose(x, (0, 2, 3, 1))
    return x


def _onehot(labels, n_cls: int) -> np.ndarray:
    """One-hot encode a 1-D label array into (N, n_cls) float32."""
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((len(labels), n_cls), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def _load_data(config):
    """Load image splits (as NHWC numpy), noisy labels, and clean truth."""
    rs = config.seed
    # cifar10n ignores noise_rate and picks labels by noise_type, so the
    # clean load must pass noise_type="clean"; for cifar10 noise_rate=0.0
    # short-circuits inject_label_noise regardless of noise_type.
    train_clean, val, test, _, _ = get_dataset(
        config.dataset, config.data_dir, noise_rate=0.0,
        noise_type="clean", random_state=rs)
    train_noisy, _, test_noisy, _, _ = get_dataset(
        config.dataset, config.data_dir, noise_rate=config.noise,
        noise_type=config.noise_type, random_state=rs)

    X_train_clean = _to_nhwc(train_clean.images.cpu().numpy())
    y_train_clean = train_clean.labels.cpu().numpy()
    y_train_noisy = train_noisy.labels.cpu().numpy()
    X_test = _to_nhwc(test_noisy.images.cpu().numpy())
    y_test_clean = test.labels.cpu().numpy()
    if int(config.n_test) < 0:
        n_test = len(X_test)  # -1 => full test split
    else:
        n_test = min(int(config.n_test), len(X_test))
    return {
        "train_clean": train_clean, "val": val, "test": test,
        "X_train_clean": X_train_clean,
        "y_train_clean": y_train_clean,
        "y_train_noisy": y_train_noisy,
        "X_test": X_test[:n_test],
        "y_test_clean": y_test_clean[:n_test],
        "y_test_noisy": test_noisy.labels.cpu().numpy()[:n_test],
        "n_test": n_test,
    }


def _encode(vae, X, device, accel, batch_size=512):
    """Batch-encode image inputs (NHWC) to reparameterised VAE latents."""
    parts = []
    dl = place_loader_local(
        tensor_loader(torch.as_tensor(X, dtype=torch.float32),
                      batch_size=batch_size, shuffle=False),
        device)
    amp = accel.autocast() if accel is not None else nullcontext()
    with torch.no_grad():
        with amp:
            for (xb,) in dl:
                mu, logvar = vae.encode(xb)
                parts.append(vae.reparameterize(mu, logvar))
    return torch.cat(parts, dim=0).detach().float().cpu().numpy()


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _save_classifier(clf, ckpt_path: Path) -> None:
    """Persist a ConvClassifier as a pickle bundle (state_dict-based)."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump({"clf": clf}, f)
    log.info(f"Saved classifier to {ckpt_path}")


def _train_or_load_conv_clf(ckpt_path: Path, X_train, y_train,
                            X_val, y_val, device, force_retrain) -> ConvClassifier:
    """Train a ConvClassifier on (X_train, y_train), or load from pickle."""
    if ckpt_path.exists() and not force_retrain:
        log.info(f"Loading classifier from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            clf = pickle.load(f)["clf"]
        return clf
    clf = train_conv_classifier(
        X_train, y_train, X_val, y_val,
        img_channels=3, img_size=32, n_classes=10, device=device)
    _save_classifier(clf, ckpt_path)
    return clf


def _compute_rec(config, out_dir: Path, data_ckpt_dir: Path, data: dict) -> list:
    """Train-or-load models and run the correction path per test query -> rec list."""
    X_train_clean = data["X_train_clean"]
    y_train_clean = data["y_train_clean"]
    y_train_noisy = data["y_train_noisy"]
    X_test = data["X_test"]
    y_test_clean = data["y_test_clean"]
    y_test_noisy = data["y_test_noisy"]
    n_test = data["n_test"]

    val = data["val"]
    X_val = _to_nhwc(val.images.cpu().numpy())
    y_val = val.labels.cpu().numpy()

    device = _select_device()
    log.info(f"Device: {device}")

    base_tag = f"{config.dataset}_{config.noise_type}{int(config.noise * 100)}"
    yhat_from_data = config.yhat_source == "data"

    # Data mode: evaluate on a held-out TRAIN slice (has both noisy yhat and
    # clean truth; removed from LSNPC training). The test set is clean-only,
    # so it is degenerate for data-source correction.
    eval_train = yhat_from_data and int(getattr(config, "eval_train_slice", 0)) > 0
    if eval_train:
        n_eval = min(int(config.eval_train_slice), len(X_train_clean))
        rng = np.random.default_rng(config.seed)
        eval_idx = rng.choice(len(X_train_clean), size=n_eval, replace=False)
        keep = np.setdiff1d(np.arange(len(X_train_clean)), eval_idx)
        X_test = np.asarray(X_train_clean)[eval_idx]
        y_test_clean = np.asarray(y_train_clean)[eval_idx]
        y_test_noisy = np.asarray(y_train_noisy)[eval_idx]
        n_test = n_eval
        X_train_clean = np.asarray(X_train_clean)[keep]
        y_train_clean = np.asarray(y_train_clean)[keep]
        y_train_noisy = np.asarray(y_train_noisy)[keep]
        log.info(
            f"data-mode eval on held-out train slice: n_eval={n_eval}, "
            f"LSNPC train={len(X_train_clean)}")

    # ── Classifiers (skipped in data mode: yhat comes from raw noisy labels) ──
    clf = None
    clf_clean = None
    if not yhat_from_data:
        clf = _train_or_load_conv_clf(                       # trained on noisy
            data_ckpt_dir / f"classifiers_{base_tag}.pkl",
            X_train_clean, y_train_noisy, X_val, y_val, device,
            config.force_retrain)
        clf_clean = _train_or_load_conv_clf(                 # trained on clean
            data_ckpt_dir / f"classifiers_clean_{base_tag}.pkl",
            X_train_clean, y_train_clean, X_val, y_val, device,
            config.force_retrain)
    else:
        log.info("yhat_source=data: training no classifier; correction conditions on raw noisy labels")

    # ── Accelerator + universal plausibility VAE (LSNPC's vae, image) ──
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mp = "bf16"
    elif torch.cuda.is_available():
        mp = "fp16"
    else:
        mp = "no"
    accel = Accelerator(mixed_precision=mp)
    device = accel.device
    log.info(f"Using Accelerator (device={device}, mp={mp})")

    vae = get_or_train(
        dataset=config.dataset, noise_type=config.noise_type,
        noise_rate=config.noise, X_train=X_train_clean, config=config,
        device=device, accel=accel, ckpt_dir=data_ckpt_dir,
        force_retrain=config.force_retrain, latent_dim=config.latent_dim,
        )

    # ── Predictive conditioning + query classes ──
    if yhat_from_data:
        n_cls = int(y_train_clean.max()) + 1
        c_soft_test = _onehot(y_test_noisy, n_cls)
        query_classes = y_test_noisy.astype(np.int64)
        c_soft_train = _onehot(y_train_noisy, n_cls)
    else:
        c_soft_test = build_predictive_conditioning(clf, X_test, mode=config.conditioning_mode)
        query_classes = clf.predict(X_test).astype(np.int64)
        c_soft_train = build_predictive_conditioning(clf, X_train_clean, mode=config.conditioning_mode)

    # ── Stage-1 LSNPC (image classification path) ──
    config.lsnpc_image_data = True
    config.lsnpc_img_channels = 3
    config.lsnpc_img_size = 32
    z_train, z_test, c_corr_train, c_corr_test, lsnpc, _clean_set_01_loss, diag = run_lsnpc_stage1(
        vae, X_train_clean, y_train_noisy, X_test, n_test,
        clf, c_soft_train, c_soft_test, y_train_clean,
        config, device, accel,
        y_test_clean=y_test_clean,
        y_test_noisy=y_test_noisy,
        yhat_source=config.yhat_source,
        # Data mode: use the (clean) test data as the pseudo-clean pool.
        pseudo_clean=(data["X_test"], data["y_test_clean"]) if yhat_from_data else None,
        val_clean=None,  # clean set drawn from X_train instead
        ckpt_dir=data_ckpt_dir,
    )
    z_test = _to_np(z_test)
    c_corr_test = _to_np(c_corr_test)

    # VAE latents for the decoder-amplification baseline.
    z_vae = _encode(vae, X_test, device, accel)

    # ── Correction path: z0 = z_test[i] (stage-1 corrected latent),
    # corrected_cls = argmax(c_corr_test[i]); decode() returns NHWC ──
    corrected_preds = np.argmax(c_corr_test, axis=-1)
    _bs = max(1, int(getattr(config, "batch_size", 256)))
    idx_dl = place_loader_local(
        tensor_loader(torch.arange(n_test), batch_size=_bs, shuffle=False),
        device)
    rec = []
    with torch.no_grad():
        for (bidx,) in idx_dl:
            idx = bidx.cpu().numpy()
            x_i = X_test[idx]
            yhat = query_classes[idx]
            ystar = y_test_clean[idx]
            z0 = torch.as_tensor(z_test[idx], device=device, dtype=torch.float32)
            x_star = np.asarray(lsnpc.decode(z0).detach().cpu())
            f_noisy_star = (clf.predict(x_star, batch_size=_bs)
                            if clf is not None else yhat.copy())
            f_clean_star = (clf_clean.predict(x_star, batch_size=_bs)
                            if clf_clean is not None
                            else -np.ones(len(idx), dtype=np.int64))
            corrected_cls = corrected_preds[idx]
            mech_flip = (corrected_cls != yhat)
            flat = x_star.reshape(len(idx), -1)
            cost = np.linalg.norm(flat - x_i.reshape(len(idx), -1), axis=1)
            z_vae_i = z_vae[idx]
            lat_shift = np.linalg.norm(z_test[idx] - z_vae_i, axis=1)
            x_zvae = np.asarray(lsnpc.decode(
                torch.as_tensor(z_vae_i, device=device, dtype=torch.float32)
            ).detach().cpu())
            dec_amp = np.linalg.norm(
                flat - x_zvae.reshape(len(idx), -1), axis=1) / (lat_shift + 1e-9)
            on_manifold_err = cost  # image path: recon error == L2 pixel distance

            # E2' random control skipped (not part of core inference);
            # field kept for schema compatibility.
            random_flip_rate = np.zeros(len(idx), dtype=np.float32)

            for j in range(len(idx)):
                rec.append({
                    "query": int(idx[j]),
                    "y_noisy": int(yhat[j]), "y_clean": int(ystar[j]),
                    "mech_flip": int(mech_flip[j]),
                    "corrected_cls": int(corrected_cls[j]),
                    "f_noisy_star": int(f_noisy_star[j]),
                    "f_clean_star": int(f_clean_star[j]),
                    "noisy_flip": int(f_noisy_star[j] != yhat[j]),
                    "clean_flip_to_truth": int(f_clean_star[j] == ystar[j]),
                    "cost": float(cost[j]), "lat_shift": float(lat_shift[j]),
                    "dec_amp": float(dec_amp[j]),
                    "on_manifold_err": float(on_manifold_err[j]),
                    "na_cost": float("nan"),
                    "random_flip_rate": float(random_flip_rate[j]),
                    "is_misclassified": int(yhat[j] != ystar[j]),
                })
    return rec


def _nanmean_skip_nan(vals) -> float:
    """NaN-mean over non-NaN entries; NaN if empty (no RuntimeWarning)."""
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=np.float64)
    return float(v.mean()) if v.size else float("nan")


def _aggregate(rec: list, config) -> dict:
    """Compute all E1-E5 aggregate metrics from the per-query records."""
    n = len(rec)
    def frac(key, sub=None):
        rows = [r for r in rec if sub(r)] if sub else rec
        return (sum(r[key] for r in rows) / len(rows)) if rows else float("nan")

    return {
        "dataset": config.dataset, "noise": config.noise,
        "noise_type": config.noise_type, "seed": config.seed, "n_test": n,
        "noisy_flip_rate": frac("noisy_flip"),
        "clean_flip_to_truth_rate": frac("clean_flip_to_truth"),
        "mechanism_flip_rate": frac("mech_flip"),
        "cf_split": frac("mech_flip"),
        "sf_split": 1.0 - frac("mech_flip"),
        "mech_flip_when_misclf": frac("mech_flip", lambda r: r["is_misclassified"]),
        "mech_flip_when_correct": frac("mech_flip", lambda r: not r["is_misclassified"]),
        "clean_flip_to_truth_when_misclf": frac("clean_flip_to_truth", lambda r: r["is_misclassified"]),
        "mean_cost": float(np.mean([r["cost"] for r in rec])),
        "mean_lat_shift": float(np.mean([r["lat_shift"] for r in rec])),
        "mean_dec_amp": float(np.mean([r["dec_amp"] for r in rec])),
        "random_flip_overall": frac("random_flip_rate"),
        "random_flip_on_misclf": frac("random_flip_rate", lambda r: r["is_misclassified"]),
        "learned_flip_on_misclf": frac("clean_flip_to_truth", lambda r: r["is_misclassified"]),
        "e2p_ratio_misclf": (
            frac("clean_flip_to_truth", lambda r: r["is_misclassified"])
            / max(1e-9, frac("random_flip_rate", lambda r: r["is_misclassified"]))
            if frac("random_flip_rate", lambda r: r["is_misclassified"]) > 0 else float("nan")),
        "mean_on_manifold_err": float(np.mean([r["on_manifold_err"] for r in rec])),
        "mean_na_cost": _nanmean_skip_nan([r["na_cost"] for r in rec]),
        "na_cost_on_misclf": _nanmean_skip_nan(
            [r["na_cost"] for r in rec if r["is_misclassified"]]),
        "misclassified_frac": frac("is_misclassified"),
        "e5_clean_flip_to_truth": frac("clean_flip_to_truth", lambda r: r["is_misclassified"]),
        "e5_noisy_flip_away": frac("noisy_flip", lambda r: r["is_misclassified"]),
    }


E_METRICS = {
    "E1": ["noisy_flip_rate", "clean_flip_to_truth_rate", "mechanism_flip_rate",
           "cf_split", "sf_split", "mech_flip_when_misclf",
           "mech_flip_when_correct", "clean_flip_to_truth_when_misclf"],
    "E2": ["mean_cost", "mean_lat_shift", "mean_dec_amp",
           "random_flip_on_misclf", "learned_flip_on_misclf", "e2p_ratio_misclf"],
    "E3": ["mean_on_manifold_err"],
    "E4": ["mean_na_cost", "na_cost_on_misclf"],
    "E5": ["misclassified_frac", "e5_clean_flip_to_truth", "e5_noisy_flip_away",
           "learned_flip_on_misclf", "random_flip_on_misclf", "e2p_ratio_misclf",
           "clean_flip_to_truth_rate", "noisy_flip_rate"],
}


def _emit(experiment: str, agg: dict, out_dir: Path, config) -> None:
    """Print (and save) the requested experiment's aggregate metrics."""
    out = out_dir / f"recourse_{_cell_key(config)}_{experiment}.json"
    cfg_snapshot = {
        "dataset": config.dataset, "noise": config.noise,
        "noise_type": config.noise_type, "seed": config.seed,
        "n_test": config.n_test, "latent_dim": config.latent_dim,
        "conditioning_mode": config.conditioning_mode,
        "lsnpc_beta": config.lsnpc_beta, "use_semi": config.use_semi,
    }
    exps = [experiment] if experiment != "ALL" else ["E1", "E2", "E3", "E4", "E5"]
    payload = {"args": cfg_snapshot, "aggregate": agg}
    for exp in exps:
        keys = E_METRICS[exp]
        sub = {k: agg[k] for k in keys}
        payload[exp] = sub
        log.info(f"[{exp}] " + "  ".join(f"{k}={v:.4f}" for k, v in sub.items() if isinstance(v, float)))
    out.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote {out}")


def main() -> None:
    # Parse --experiment out of sys.argv, then hand the rest to
    # ExperimentConfig.from_args() (which reads sys.argv).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--experiment", default="ALL")
    known, leftovers = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + leftovers
    experiment = known.experiment.upper()
    if experiment not in EXPERIMENTS:
        pre.error(f"unknown --experiment {experiment!r}; use E1..E5 or ALL")

    config = ExperimentConfig.from_args()
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_ckpt_dir = out_dir / "ckpt" / f"{config.dataset}_{config.noise_type}{int(config.noise * 100)}_seed{config.seed}"
    data_ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(config)
    cache_path = out_dir / f"rec_cache_{_cell_key(config)}_nt{data['n_test']}.json"
    if cache_path.exists():
        log.info(f"Loading cached correction path from {cache_path}")
        rec = json.loads(cache_path.read_text())
    else:
        rec = _compute_rec(config, out_dir, data_ckpt_dir, data)
        cache_path.write_text(json.dumps(rec))
        log.info(f"Cached correction path to {cache_path}")

    agg = _aggregate(rec, config)
    _emit(experiment, agg, out_dir, config)


if __name__ == "__main__":
    main()
