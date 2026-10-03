"""X6-image: correction-method ablation on the CIFAR cells (S4 scope check).

Mirrors scripts/analyze_x6_ablation.py for the image harness:
  STANDARD  z0 = sample_corrected_latent(x, z_vae, yhat)   (LSNPC as-is)
  ORACLE    z0 = sample_corrected_latent(x, z_vae, y*)     (truth-conditioned)
  DIRECT    label from corrected_conditioning(x, z_vae)     (no shift at all)

Key question (Branch C vs B2 on multiclass images): on the misclassified
slice, does ORACLE rescue the confident-wrong queries that STANDARD
fails? If the shift itself adds nothing over DIRECT, the multiclass arm
replicates the tabular finding; if ORACLE succeeds where STANDARD fails,
the conditioning is the bottleneck on images too.

Load-only (all checkpoints cached by the S4 grid). One GPU process.
Run from repo root:  python -m scripts.analyze_x6_image
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from data_process.base import get_dataset
from experiments.arguments import ExperimentConfig
from experiments.lsnpc_stage1 import run_lsnpc_stage1
from scripts.analyze_correction_recourse_image import (
    _encode,
    _to_nhwc,
    _train_or_load_conv_clf,
)
from trainers.plausibility_vae import get_or_train as get_or_train_vae
from utils.conditioning import build_predictive_conditioning

ROOT = Path(__file__).resolve().parent.parent
RECOURSE_DIR = ROOT / "results" / "recourse"
DATA_DIR = ROOT / "data"

CELLS = []
for ds, nts in [
    ("cifar10", [("symmetric", 0.0), ("symmetric", 0.2), ("symmetric", 0.4)]),
    ("cifar10n", [("symmetric", 0.2), ("worse", 0.4)]),
]:
    for nt, noise in nts:
        for seed in (42, 43, 44):
            CELLS.append((ds, nt, noise, seed))


def _load_models(config):
    """Mirror analyze_correction_recourse_image model loading (load-only)."""
    ds, ntype, noise, seed = config
    sys.argv = ["x6img"] + [
        "--dataset", ds, "--noise", str(noise),
        "--noise-type", ntype, "--seed", str(seed),
        "--conditioning-mode", "posterior",
        "--use-semi", "--lsnpc-beta", "0.5",
        "--vae-epochs", "200", "--lsnpc-epochs", "30",
        "--n-test", "-1", "--output-dir", "results/recourse",
    ]
    cfg = ExperimentConfig.from_args()
    cfg.validate()

    rs = seed
    train_clean, val, test, _, _ = get_dataset(
        ds, str(DATA_DIR), noise_rate=0.0, noise_type="clean", random_state=rs)
    train_noisy, _, test_noisy, _, _ = get_dataset(
        ds, str(DATA_DIR), noise_rate=noise, noise_type=ntype, random_state=rs)

    X_train_clean = _to_nhwc(train_clean.images.cpu().numpy())
    y_train_clean = train_clean.labels.cpu().numpy()
    y_train_noisy = train_noisy.labels.cpu().numpy()
    X_test = _to_nhwc(test_noisy.images.cpu().numpy())
    y_test_clean = test.labels.cpu().numpy()
    X_val = _to_nhwc(val.images.cpu().numpy())
    y_val = val.labels.cpu().numpy()

    accel = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    device = accel.device

    base_tag = f"{ds}_{ntype}{int(noise * 100)}"
    ckpt_dir = RECOURSE_DIR / "ckpt" / f"{ds}_{ntype}{int(noise * 100)}_seed{seed}"

    clf = _train_or_load_conv_clf(
        ckpt_dir / f"classifiers_{base_tag}.pkl",
        X_train_clean, y_train_noisy, X_val, y_val, device, force_retrain=False)

    vae = get_or_train_vae(
        dataset=ds, noise_type=ntype, noise_rate=noise, X_train=X_train_clean,
        config=cfg, device=device, accel=accel,
        ckpt_dir=ckpt_dir, force_retrain=False, latent_dim=cfg.latent_dim,
        )

    c_soft_train = build_predictive_conditioning(clf, X_train_clean, mode="posterior")
    c_soft_test = build_predictive_conditioning(clf, X_test, mode="posterior")
    query_classes = clf.predict(X_test).astype(np.int64)

    cfg.lsnpc_image_data = True
    cfg.lsnpc_img_channels = 3
    cfg.lsnpc_img_size = 32
    z_train, z_test, c_corr_train, c_corr_test, lsnpc, _clean_set_01_loss, _diag = \
        run_lsnpc_stage1(
            vae, X_train_clean, y_train_noisy, X_test, len(X_test),
            clf, c_soft_train, c_soft_test, y_train_clean,
            cfg, device, accel,
            y_test_clean=y_test_clean, val_clean=None,
            ckpt_dir=ckpt_dir)

    z_vae = _encode(vae, X_test, device, accel)
    return {
        "X_test": X_test, "y_test_clean": y_test_clean,
        "query_classes": query_classes, "lsnpc": lsnpc,
        "z_vae": z_vae, "clf": clf,
    }


def main() -> None:
    out = {}
    for config in CELLS:
        ds, ntype, noise, seed = config
        print(f"=== X6-image {ds} {ntype}{int(noise * 100)} seed{seed} ===", flush=True)
        M = _load_models(config)
        lsnpc, z_vae = M["lsnpc"], M["z_vae"]
        X_test, y_true = M["X_test"], M["y_test_clean"]
        yhat = M["query_classes"]
        n = len(X_test)

        device = next(lsnpc.parameters()).device
        rows = []
        with torch.no_grad():
            for i in range(n):
                # The LSNPC image path (noisy_encoder, label head) expects
                # NCHW 4D; the cache/analyzer convention is NHWC numpy, so
                # permute here (mirrors trainers/lsnpc.py X_train_4d).
                x_i = torch.as_tensor(X_test[i:i + 1], device=device,
                                      dtype=torch.float32).permute(0, 3, 1, 2)
                z_vae_i = torch.as_tensor(z_vae[i:i + 1], device=device,
                                          dtype=torch.float32)
                yhat_i = torch.as_tensor([yhat[i]], device=device)
                ytrue_i = torch.as_tensor([y_true[i]], device=device)

                z0_hat = lsnpc.sample_corrected_latent(x_i, z_vae_i, yhat_i)
                z0_true = lsnpc.sample_corrected_latent(x_i, z_vae_i, ytrue_i)
                c_hat = lsnpc.corrected_conditioning(x_i, z0_hat, mode="xz")
                c_true = lsnpc.corrected_conditioning(x_i, z0_true, mode="xz")
                c_dir = lsnpc.corrected_conditioning(x_i, z_vae_i, mode="xz")
                rows.append({
                    "yhat": int(yhat[i]), "ytrue": int(y_true[i]),
                    "std_cls": int(c_hat.argmax(-1).item()),
                    "oracle_cls": int(c_true.argmax(-1).item()),
                    "direct_cls": int(c_dir.argmax(-1).item()),
                })
        rec = rows
        yhat_a = np.array([r["yhat"] for r in rec])
        ytrue_a = np.array([r["ytrue"] for r in rec])
        std = np.array([r["std_cls"] for r in rec])
        ora = np.array([r["oracle_cls"] for r in rec])
        dire = np.array([r["direct_cls"] for r in rec])
        sl = yhat_a != ytrue_a

        def rate(mask, pred):
            return float(np.mean(pred[mask] == ytrue_a[mask])) if mask.sum() else float("nan")

        clf_probs = M["clf"].predict_proba(X_test)
        entropy = -np.sum(clf_probs * np.log(np.clip(clf_probs, 1e-12, 1.0)), axis=1)
        sl_low_ent = sl & (entropy < np.median(entropy[sl]))    # confident-wrong
        sl_high_ent = sl & (entropy >= np.median(entropy[sl]))  # near-boundary

        cell = {
            "cell": f"{ds}_{ntype}{int(noise * 100)}_seed{seed}",
            "n_test": n, "slice_size": int(sl.sum()),
            "success_std_all": rate(np.ones(n, bool), std),
            "success_oracle_all": rate(np.ones(n, bool), ora),
            "success_direct_all": rate(np.ones(n, bool), dire),
            "success_std_slice": rate(sl, std),
            "success_oracle_slice": rate(sl, ora),
            "success_direct_slice": rate(sl, dire),
            "success_std_confident": rate(sl_low_ent, std),
            "success_oracle_confident": rate(sl_low_ent, ora),
            "success_direct_confident": rate(sl_low_ent, dire),
            "success_std_boundary": rate(sl_high_ent, std),
            "success_oracle_boundary": rate(sl_high_ent, ora),
            "success_direct_boundary": rate(sl_high_ent, dire),
        }
        out[cell["cell"]] = cell
        print(f"  slice={cell['slice_size']} std={cell['success_std_slice']:.3f} "
              f"ora={cell['success_oracle_slice']:.3f} "
              f"dir={cell['success_direct_slice']:.3f} "
              f"| confident: std={cell['success_std_confident']:.3f} "
              f"ora={cell['success_oracle_confident']:.3f} "
              f"dir={cell['success_direct_confident']:.3f}", flush=True)

    (RECOURSE_DIR / "x6_image.json").write_text(json.dumps(out, indent=1))
    print("saved results/recourse/x6_image.json")


if __name__ == "__main__":
    main()
