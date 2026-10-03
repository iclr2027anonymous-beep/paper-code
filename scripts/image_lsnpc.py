"""LSNPC label correction on CIFAR-10N / CIFAR-100N images (data-level).

Mirror of ``scripts/text_lsnpc_sst2.py`` for images. The frozen RN50
feature embedding (2048-d) is the latent ``z`` — the same role the mpnet
sentence embedding plays in the text pipeline. Labels come from the DATA:
the real human noisy labels (aggre ~9% / fine ~40%) are the observed
stream yhat; the ``clean_label`` view is the oracle. No classifier is
involved anywhere (label source = data), no synthetic injection (the
noise is real).

Splits mirror the text design exactly: eval slice (data-level protocol
scores vs clean oracle), clean set (clean labels, semi-supervised
supervision), val split (LSNPC checkpoint selection), train on the rest.

Usage: accelerate launch --mixed_precision bf16 -m scripts.image_lsnpc \
    --dataset cifar10n [--epochs 10 --beta 0.5 --seed 42]
"""
from __future__ import annotations

from utils.paths import project_path

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,  # override any library-set root config so [LSNPC] lines show
)

from accelerate import Accelerator

from data_process.cifar10n import _load_cifar10n
from data_process.cifar100n import _load_cifar100n
from data_process.dopanim import _load_dopanim, _load_dopanim_pickle, load_clean_test_pool
from data_process.eurosat import load_eurosat_pair
from data_process.animal10n import load_animal10n_pair, load_animal10n_clean_pool
from experiments.arguments import ExperimentConfig
from experiments.protocol import (
    SCORE_KEYS,
    compute_protocol_scores,
    multiclass_f1,
    multiclass_precision,
    multiclass_recall,
)
from experiments.deployed_frame import emit_frame_report
from models.rn50_features import IdentityFeatureEncoder, RN50FeatureEncoder
from scripts.lsnpc_ckpt import bundled_build_kwargs, load_bundle, save_bundle
from utils.batching import place_loader_local, tensor_loader
from trainers.lsnpc import (
    LSNPCTrainer, _encode_vae_latents, _encoder_liveness,
)
from experiments.lsnpc_stage1 import _resolve_lsnpc_eta, resolve_clean_rows
from trainers.plausibility_vae import get_or_train as get_or_train_vae

DATA_ROOT = Path(project_path('data'))

# Real-noise datasets select their human label set via `noise_type`; the
# loader aliases 'symmetric'/'noisy' to the canonical published aggregate.
# Synthetic benchmarks (eurosat) take symmetric/IDN noise at a rate > 0.
_REAL_NOISE_DATASETS = ("cifar10n", "cifar100n", "dopanim", "animal10n")
_REAL_VIEW_ALIAS = {
    "cifar10n": {"symmetric": "aggre", "pairflip": "aggre", "aggre": "aggre",
                 "worse": "worse"},
    "cifar100n": {"symmetric": "fine", "noisy": "fine", "pairflip": "fine",
                  "fine": "fine", "coarse": "coarse"},
    "dopanim": {"symmetric": "symmetric", "noisy": "symmetric",
                "rand": "symmetric"},
    # animal10n ships one label set: the native human annotations. The aliases
    # exist only so the real-noise lookup succeeds; no view is selected.
    "animal10n": {"symmetric": "native", "noisy": "native", "native": "native"},
}
_DEFAULT_VIEW = {"cifar10n": "aggre", "cifar100n": "fine",
                 "dopanim": "symmetric", "eurosat": "symmetric",
                 "animal10n": "native"}


def _encode_pixel_latents(vae, X_np: np.ndarray, device: str, bs: int = 256) -> np.ndarray:
    """Deterministic z_vae latents (mu head only) for NHWC pixel inputs."""
    return _encode_vae_latents(vae, X_np, device, batch_size=bs).cpu().numpy()


def load_image_pair(dataset: str, noise_type: str, noise_rate: float = 0.0,
                     pool_full: bool = False):
    """Row-aligned (X, y_clean, y_noisy) for cifar10n / cifar100n / eurosat.

    ``noise_type`` selects the label set:
      - cifar10n: 'aggre' (via 'symmetric'), 'worse', 'clean';
      - cifar100n: 'fine' (via 'symmetric'/'noisy'), 'coarse', 'clean';
      - eurosat: synthetic noise (``noise_type`` symmetric/idn at
        ``noise_rate`` > 0 injected once over the full label array).
    The images are shared across the clean and noisy views (same split
    permutation), so clean and noisy labels align row-for-row.

    ``pool_full=True`` (dopanim only): bypass the loader's internal 85/15
    permutation and return the FULL annotated pool (10,484 rows) in raw
    pickle order, CIFAR-normalized — the Set E dopanim carve (E2: annota-
    tions live on the train split only; clean set carved from the clean
    test pool).
    """
    if dataset == "cifar10n":
        load = _load_cifar10n
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100n":
        load = _load_cifar100n
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    elif dataset == "eurosat":
        X, y_noisy, y_clean, X_test, y_test, mean, std = load_eurosat_pair(
            str(DATA_ROOT), noise_rate=float(noise_rate),
            noise_type=noise_type, random_state=42)
        return X, y_noisy, y_clean, y_test, mean, std
    elif dataset == "animal10n":
        # Real human noise with no train-side oracle: y_clean is the -1 sentinel
        # and the clean pool is carved from the clean test split (see the loader).
        X, y_noisy, y_clean, X_test, y_test, mean, std = load_animal10n_pair(
            str(DATA_ROOT), noise_rate=0.0, noise_type=noise_type, random_state=42)
        return X, y_noisy, y_clean, y_test, mean, std
    elif dataset == "dopanim":
        if pool_full:
            # Set E dopanim carve: FULL annotated pool in raw pickle order
            # (no 85/15 shuffle — the bundle split indices then index the
            # raw 10,484-row pool that e_common.load_dopanim_view replays);
            # the held-out clean test pool supplies the clean-set source.
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            mean_t = torch.tensor(mean).view(1, 3, 1, 1)
            std_t = torch.tensor(std).view(1, 3, 1, 1)
            Xp_u8, y_clean_p, y_noisy_p = _load_dopanim_pickle(str(DATA_ROOT))
            x = torch.from_numpy(np.asarray(Xp_u8)).float() / 255.0
            Xn = ((x - mean_t) / std_t).numpy().astype(np.float32)
            Xc_pool, yc_test_pool = load_clean_test_pool(str(DATA_ROOT))
            return (Xn,
                    np.asarray(y_noisy_p).ravel(),
                    np.asarray(y_clean_p).ravel(),
                    np.asarray(yc_test_pool).ravel(),
                    mean, std)
        load = _load_dopanim
        # CIFAR constants: _pretrained_preprocess denormalizes by these, so
        # any pixel-mode input must use the shared CIFAR normalization.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError(dataset)

    tr_n, va_n, te_n, _ = load(str(DATA_ROOT), noise_rate=float(noise_rate),
                               noise_type=noise_type, random_state=42)
    tr_c, va_c, te_c, _ = load(str(DATA_ROOT), noise_rate=0.0,
                               noise_type="clean", random_state=42)
    Xn = np.asarray(tr_n.features)
    Xc = np.asarray(tr_c.features)
    assert Xn.shape == Xc.shape and (Xn == Xc).all(), "views not row-aligned"
    return (np.asarray(tr_n.features), np.asarray(tr_n.targets).ravel(),
            np.asarray(tr_c.targets).ravel(), np.asarray(te_c.targets).ravel(),
            mean, std)


def build_config(seed, epochs, beta, batch, clean_set_size, eval_slice,
                 latent_dim, dataset, n_classes, trainable=False,
                 img_size=32, eta=None, backbone="resnet50", pixel=False,
                 lsnpc_lr=5e-4, hidden_dim=256, n_blocks=4, focal_gamma=0.0,
                 focal_alpha=None, head="concat", backbone_lr_scale=1.0,
                 embed_dim=128,
                 correction_cond="none", shared_yhat_embed=False,
                 correction_input="gated"):
    sys.argv = ["image"] + [
        "--dataset", dataset, "--noise", "0.0", "--noise-type", "symmetric",
        "--seed", str(seed), "--conditioning-mode", "posterior",
        "--use-semi", "--lsnpc-beta", str(beta), "--lsnpc-lr", str(lsnpc_lr),
        "--clean-set-size", str(clean_set_size), "--lsnpc-epochs", str(epochs),
        "--n-test", "-1", "--yhat-source", "data",
        "--eval-train-slice", str(eval_slice),
        "--encoder-backbone", backbone, "--batch-size", str(batch),
        "--output-dir", f"/tmp/image_{dataset}_s{seed}_e{epochs}_b{beta}_cs{clean_set_size}",
        "--latent-dim", str(latent_dim), "--hidden-dim", str(hidden_dim),
        "--n-blocks", str(n_blocks), "--iw-samples", "5",
        "--focal-gamma", str(focal_gamma),
    ]
    if focal_alpha is not None:
        sys.argv += ["--focal-alpha", str(focal_alpha)]
    if eta is not None:
        sys.argv += ["--lsnpc-eta", str(eta)]
    if backbone_lr_scale != 1.0:
        sys.argv += ["--lsnpc-backbone-lr-scale", str(backbone_lr_scale)]
    if head != "concat":
        sys.argv += ["--lsnpc-head", str(head)]
    if embed_dim != 128:
        sys.argv += ["--lsnpc-embed-dim", str(embed_dim)]
    if correction_cond != "none":
        sys.argv += ["--correction-cond", str(correction_cond)]
    if shared_yhat_embed:
        sys.argv += ["--shared-yhat-embed"]
    if correction_input != "gated":
        sys.argv += ["--correction-input", str(correction_input)]
    # ── Pixel mode (trainable or frozen backbone): small-latent image mode ──
    # The backbone IS the image encoder; its mu head maps the embedding DOWN
    # to latent_dim, so the correction space stays small rather than
    # embedding-sized. Input is raw NHWC CIFAR pixels.
    if pixel:
        sys.argv += ["--lsnpc-image-data", "true", "--lsnpc-img-size", str(img_size),
                     "--lsnpc-img-channels", "3"]
        if trainable:
            sys.argv += ["--no-freeze-backbone"]
    config = ExperimentConfig.from_args()
    config.validate()
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    if not pixel:
        config.lsnpc_image_data = False  # embedding-space: features are the latent
    return config


def _lsnpc_flag_tokens(args) -> dict[str, str]:
    """Name tokens for the objective/structure flags, resolved in one place.

    Every token is the empty string at its default, so a default run's name is
    unchanged. Both the bundle filename and the scratch output directory are
    built from these, because the two collided once: with no tokens in the
    directory, a seed-42 ``result.json`` was overwritten by a different arm of
    the same cell (two arms of one cell differing only in a flag that had no
    token).
    """
    return dict(
        focal=(f"_fg{args.focal_gamma:g}" if args.focal_gamma else ""),
        falpha=(f"_fa{args.focal_alpha:g}"
                if args.focal_alpha is not None else ""),
        head=(f"_h{args.lsnpc_head}" if args.lsnpc_head != "concat" else ""),
        embed=(f"_ed{args.lsnpc_embed_dim}"
               if args.lsnpc_embed_dim != 128 else ""),
        ccond=("" if args.correction_cond == "none"
               else f"_cc{args.correction_cond}"),
        syhat=("_sy" if getattr(args, "shared_yhat_embed", False) else ""),
        civ=("" if getattr(args, "correction_input", "gated") == "gated"
             else f"_ci{args.correction_input}"),
    )


def _lsnpc_run_name(config, args, trainable: bool, pixel: bool,
                    cell_tag: str | None = None) -> str:
    """Cell id + flag tokens for a run, shared by the bundle name and the
    scratch output directory (the ``lsnpc_``/``image_`` prefix differs)."""
    t = _lsnpc_flag_tokens(args)
    return (f"{args.dataset}_s{args.seed}_e{args.epochs}_b{args.beta}"
            f"_cs{args.clean_set_size}"
            + t["focal"] + t["falpha"]
            + (f"_eta{args.eta}" if args.eta is not None else "")
            + (f"_{args.backbone}" if args.backbone != "resnet50" else "")
            + ("_train" if trainable else "")
            # No BatchNorm token: the norm layers follow ``trainable`` exactly
            # (frozen backbone = frozen pretrained stats, trainable = stats
            # update), so ``_train`` already pins the recipe.
            + t["head"] + t["embed"] + t["ccond"]
            + t["syhat"] + t["civ"]
            + (f"_sz{config.lsnpc_img_size}" if pixel else "")
            + (f"_{cell_tag}" if cell_tag else ""))


def _lsnpc_ckpt_name(config, args, trainable: bool, pixel: bool,
                   cell_tag: str | None = None) -> str:
    """Bundle filename for a trained image LSNPC (shared by save & emit-only)."""
    return "lsnpc_" + _lsnpc_run_name(config, args, trainable, pixel, cell_tag) + ".pt"


def _lsnpc_output_dir(config, args, trainable: bool, pixel: bool,
                      cell_tag: str | None = None) -> str:
    """Scratch directory for ``result.json`` / the per-query npz.

    Same tokens as the bundle name on purpose, so two arms of one cell cannot
    write over each other's metrics (see ``_lsnpc_flag_tokens``).
    """
    return ("/tmp/image_"
            + _lsnpc_run_name(config, args, trainable, pixel, cell_tag))


def _save_lsnpc_ckpt(config, model, n_classes, x_dim, splits, args, trainable,
                     pixel, val_err_h, cell_tag: str | None = None) -> Path:
    """Save the trained LSNPC to results/ckpt/lsnpc/ for re-use (Set E)."""
    name = _lsnpc_ckpt_name(config, args, trainable, pixel, cell_tag)
    split_seed = (args.split_seed if args.split_seed is not None
                  else args.seed)
    run = dict(dataset=args.dataset, seed=args.seed, epochs=args.epochs,
               beta=args.beta, eta=args.eta, clean_set_size=args.clean_set_size,
               eval_slice=args.eval_slice, val_size=args.val_size,
               backbone=args.backbone, trainable=trainable, pixel=pixel,
               batch_size=args.batch_size, lsnpc_lr=args.lsnpc_lr,
               hidden_size=args.hidden_size, n_blocks=args.n_blocks,
               split_seed=int(split_seed),
               allow_large_eval=bool(args.allow_large_eval),
               pool_full=bool(args.pool_full))
    run["noise"] = float(args.noise)
    run["noise_type"] = str(args.noise_type)
    run["rob_k"] = int(args.rob_k)
    run["lsnpc_head"] = str(args.lsnpc_head)
    run["lsnpc_embed_dim"] = int(args.lsnpc_embed_dim)
    run["lsnpc_correction_cond"] = str(args.correction_cond)
    run["cell_tag"] = str(cell_tag)
    out = save_bundle(
        Path("results/ckpt/lsnpc/image") / name, model,
        arch={"build_kwargs": bundled_build_kwargs(
            config, x_dim=x_dim, n_classes=n_classes)},
        splits=splits, run=run, val_err_h=val_err_h)
    print(f"[ckpt] LSNPC bundle saved: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10n",
                    choices=["cifar10n", "cifar100n", "dopanim", "eurosat", "animal10n"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="RNG seed for the eval/val/cs split carves. Default: "
                         "--seed. Set E dopanim fixes ONE permutation across "
                         "corrector seeds (split-seed 42) so corrector-level "
                         "variance is not confounded with stream differences.")
    ap.add_argument("--pool-full", action="store_true",
                    help="dopanim only: use the FULL 10,484-row annotated "
                         "pool in raw order (Set E dopanim carve) instead of "
                         # ``%%``: argparse formats every help string with ``%``.
                         "the loader's 85%% train subset.")
    ap.add_argument("--allow-large-eval", action="store_true",
                    help="dopanim only: lift the eval carve n//4 cap (Set E "
                         "dopanim carve: eval >= 3000, val >= 2500).")
    ap.add_argument("--eval-slice", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--clean-set-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lsnpc-lr", type=float, default=5e-4,
                    help="LSNPC stage-1 learning rate (default 5e-4)")
    ap.add_argument("--lsnpc-backbone-lr-scale", type=float, default=1.0,
                    help="LR multiplier for the image backbone (1.0 = same as "
                         "the correction heads; small values protect the "
                         "pretrained features while fine-tuning).")
    ap.add_argument("--hidden-size", type=int, default=256,
                    help="MLP hidden width (maps to --hidden-dim; default 256)")
    ap.add_argument("--n-blocks", "--n_blocks", type=int, default=4,
                    help="MLP trunk blocks (default 4)")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="focal-loss focusing parameter on the label terms "
                         "(0 = plain cross-entropy)")
    ap.add_argument("--focal-alpha", type=float, default=None,
                    help="focal-loss alpha balance of the one-vs-all terms; "
                         "omit for unweighted terms")
    ap.add_argument("--lsnpc-head", choices=["concat", "gate"],
                    default="concat",
                    help="label-head structure: 'concat' (original) or 'gate' "
                         "(one trunk per input, mixed by a learned gate)")
    ap.add_argument("--lsnpc-embed-dim", type=int, default=128,
                    help="width of the trainable projection from the "
                         "backbone's pooled feature (RN50: 2048-d) into the "
                         "LSNPC image embedding; the raw backbone feature is "
                         "never fed to the head")
    ap.add_argument("--correction-cond",
                    choices=["none", "x", "x_yhat", "yhat"], default="none",
                    help="Inputs of the correction map q(z | ẑ, ·): 'none' "
                         "(the VAE latent of x), 'x', 'x_yhat', or 'yhat' "
                         "(no x: the blend target is a learned label embedding).")
    ap.add_argument("--shared-yhat-embed", action="store_true",
                    help="Shared label-embedding library (#4): one learnable "
                         "per-class embedding feeds both the posterior's label "
                         "slot and the blend target.")
    ap.add_argument("--correction-input", choices=["gated", "concat"],
                    default="gated",
                    help="Correction map input: 'gated' (blend) or 'concat' "
                         "(a trunk over cat([ẑ, x_feat])).")
    ap.add_argument("--eta", type=float, default=None,
                    help="η (clean-set scheduling probability). Default: 0.1 "
                         "when use_semi (auto-resolution in stage-1/trainer).")
    ap.add_argument("--trainable", action="store_true",
                    help="Fine-tune the backbone: pixel-native image mode with "
                         "an UNFROZEN encoder (no-freeze-backbone).")
    ap.add_argument("--pixel", action="store_true",
                    help="Pixel-native image mode with a FROZEN backbone "
                         "(freeze_backbone default True); pair with "
                         "--backbone to choose the encoder.")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "resnet101", "swin", "swin_tiny",
                             "vit"],
                    help="Image encoder backbone for pixel mode (default resnet50).")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Synthetic noise rate for clean benchmarks "
                         "(eurosat); ignored by real-noise datasets.")
    ap.add_argument("--noise-type", default=None,
                    help="Label set: real views 'aggre'/'worse' (cifar10n), "
                         "'fine'/'coarse' (cifar100n); synthetic noise types "
                         "'symmetric'/'idn' for eurosat. Defaults per "
                         "dataset (aggre / fine / symmetric).")
    ap.add_argument("--rob-k", type=int, default=4,
                    help="Noise-robustness perturbation count K "
                         "(protocol Set A uses K=32).")
    ap.add_argument("--emit-only", action="store_true",
                    help="Skip training; load the saved ckpt bundle and only "
                         "re-emit protocol scores (e.g. plausibility reruns).")
    args = ap.parse_args()
    pixel = args.pixel or args.trainable
    trainable = args.trainable
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.allow_large_eval and args.dataset != "dopanim":
        raise SystemExit("--allow-large-eval is reserved for dopanim (Set E "
                         "corrector carve).")
    if args.pool_full and args.dataset != "dopanim":
        raise SystemExit("--pool-full is reserved for dopanim (Set E carve).")
    if args.pool_full and not (args.pixel or args.trainable):
        raise SystemExit("--pool-full requires the pixel path (dopanim E "
                         "corrector is frozen Swin @224).")

    if args.dataset not in _DEFAULT_VIEW:
        raise ValueError(f"Unknown dataset {args.dataset!r}; expected one of "
                         f"{sorted(_DEFAULT_VIEW)}")
    if args.noise_type is None:
        args.noise_type = _DEFAULT_VIEW[args.dataset]
    real = args.dataset in _REAL_NOISE_DATASETS
    if real:
        if args.noise != 0.0:
            raise ValueError(
                f"{args.dataset} carries REAL human noise — synthetic "
                f"injection is never applied (got --noise {args.noise}).")
        if args.noise_type not in _REAL_VIEW_ALIAS[args.dataset]:
            raise ValueError(
                f"{args.dataset}: unknown real label view "
                f"{args.noise_type!r}; expected one of "
                f"{sorted(_REAL_VIEW_ALIAS[args.dataset])}")
        args.noise_type = _REAL_VIEW_ALIAS[args.dataset][args.noise_type]
    else:
        if args.noise_type not in ("symmetric", "idn"):
            raise ValueError(
                f"{args.dataset} is a clean benchmark: --noise-type must be "
                f"'symmetric' or 'idn' (got {args.noise_type!r})")
        if args.noise <= 0.0:
            raise ValueError(
                f"{args.dataset}: synthetic cells need --noise > 0 "
                f"(got {args.noise}).")
    if args.rob_k < 1:
        raise ValueError(f"--rob-k must be >= 1 (got {args.rob_k})")
    # Cell tag: real views are distinct runs of the same images (aggre vs
    # worse) and synthetic rates would otherwise collide in output dirs.
    cell_tag = (args.noise_type if real
                else f"n{args.noise:g}_{args.noise_type}")
    print(f"[{args.dataset}] view={args.noise_type} noise={args.noise} "
          f"rob-K={args.rob_k} tag={cell_tag}")

    # Real human labels: data-level (no classifier, no synthetic injection).
    X, y_noisy_all, y_clean_all, y_test_clean, _, _ = load_image_pair(
        args.dataset, noise_type=args.noise_type, noise_rate=args.noise,
        pool_full=args.pool_full)
    n_classes = int(max(y_clean_all.max(), y_noisy_all.max())) + 1
    n = len(X)
    rng = np.random.default_rng(split_seed)
    ev_n = (args.eval_slice if (args.allow_large_eval
                                and args.dataset == "dopanim")
            else min(args.eval_slice, n // 4))
    eval_idx = rng.choice(n, size=ev_n, replace=False)
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

    # Special real-noise datasets (dopanim): annotations live ONLY on the
    # train split, so a clean set carved from the train pool is not
    # independent of the noise process. The held-out clean test split is
    # the clean-set source instead; train/val/eval still come from the
    # annotated pool (val/eval need the noisy labels for the protocol).
    # animal10n belongs here for the same reason: it releases no clean training
    # labels, so a train-pool carve would supervise on the -1 sentinel.
    clean_from_test = args.dataset in ("dopanim", "animal10n")
    if clean_from_test:
        if args.dataset == "animal10n":
            X_cs_pool, y_cs_pool = load_animal10n_clean_pool(str(DATA_ROOT))
        else:
            X_cs_pool, y_cs_pool = load_clean_test_pool(str(DATA_ROOT))
        n_cs_pool = len(X_cs_pool)
        N = resolve_clean_rows(args.clean_set_size, n_cs_pool)
        cs_perm = rng.permutation(n_cs_pool)
        cs_idx = cs_perm[:N]
        Xcs = X_cs_pool[cs_idx]
        ycs = y_cs_pool[cs_idx]
        tr_idx = pool  # everything left in the annotated pool is training
        print(f"[{args.dataset}] clean set carved from HELD-OUT CLEAN TEST "
              f"split: {len(Xcs)} rows (of {n_cs_pool})")
        # animal10n: checkpoint selection needs labelled validation rows, and
        # the pooled train rows carry only the -1 sentinel. Carve the validation
        # from the same clean pool, disjoint from the clean set, and leave the
        # eval slice in the pool (its audit metrics are not reportable here).
        val_from_clean = (args.dataset == "animal10n")
        if val_from_clean:
            n_vc = min(args.val_size, n_cs_pool - N)
            v_idx = cs_perm[N:N + n_vc]
            _Xv_clean, _yv_clean = X_cs_pool[v_idx], y_cs_pool[v_idx]
    else:
        N = resolve_clean_rows(args.clean_set_size, len(pool) // 2)
        cs_idx = pool[:N]
        tr_idx = pool[N:]
        Xcs, ycs = X[cs_idx], y_clean_all[cs_idx]
        print(f"[{args.dataset} {n_classes}cls] clean set carved from train "
              f"pool: {len(Xcs)} rows")

    Xe, yce = X[eval_idx], y_clean_all[eval_idx]
    Xtr, yctr = X[tr_idx], y_clean_all[tr_idx]
    yntr = y_noisy_all[tr_idx]
    if args.dataset == "animal10n":
        Xv, ycv = _Xv_clean, _yv_clean
        ynv = _yv_clean  # these rows are uncorrupted, so no noisy label exists
        print(f"[{args.dataset}] validation carved from the clean pool: "
              f"{len(Xv)} rows (disjoint of the clean set)")
    else:
        Xv, ycv = X[val_idx], y_clean_all[val_idx]
        ynv = y_noisy_all[val_idx]
    yne = y_noisy_all[eval_idx]
    print(f"[{args.dataset} {n_classes}cls] splits: train={len(Xtr)} "
          f"clean_set={len(Xcs)} val={len(Xv)} eval={len(Xe)}; "
          f"data flip frac={float((yntr != yctr).mean()):.3f} "
          + ("(real labels)" if real
             else f"(injected {args.noise_type}@{args.noise:g})"))

    # ── Input layout ────────────────────────────────────────────────
    # Loader returns NCHW CIFAR-normalized pixels. The trainer's image
    # branch expects NHWC numpy (it permutes (0,3,1,2) internally); the
    # frozen-feature path featurizes NCHW directly.
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if pixel:
        # ── Pixel-native image mode, small latent ──
        # The backbone IS the model's own image encoder (frozen unless
        # --trainable); its mu head Linear(feat -> latent_dim=64) converts the
        # embedding down to the correction latent. z_vae latents (64-d content
        # codes) come from the codebase's conv-VAE recipe (get_or_train),
        # cached per dataset and reused across seeds/clean sizes.
        latent_dim = 64
        px = X.shape[-1] if X.ndim == 4 else 32
        config = build_config(args.seed, args.epochs, args.beta, args.batch_size,
                              args.clean_set_size, args.eval_slice, latent_dim,
                              args.dataset, n_classes, trainable=trainable,
                              eta=args.eta, backbone=args.backbone, pixel=True,
                              img_size=px, lsnpc_lr=args.lsnpc_lr,
                              hidden_dim=args.hidden_size, n_blocks=args.n_blocks,
                              focal_gamma=args.focal_gamma,
                              focal_alpha=args.focal_alpha,
                              head=args.lsnpc_head,
                              embed_dim=args.lsnpc_embed_dim,
                              correction_cond=args.correction_cond,
                              shared_yhat_embed=args.shared_yhat_embed,
                              correction_input=args.correction_input)
        config.output_dir = _lsnpc_output_dir(
            config, args, trainable=trainable, pixel=True, cell_tag=cell_tag)
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"[pixel] output dir carries pixel size: sz{px}")
        print(f"[out] {config.output_dir}")

        def _nhwc(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return (x.transpose(0, 2, 3, 1) if (x.ndim == 4 and x.shape[1] == 3
                                                and x.shape[-1] != 3) else x)

        Xe_n, Xtr_n, Xcs_n, Xv_n = (_nhwc(a) for a in (Xe, Xtr, Xcs, Xv))
        print(f"[pixel {args.backbone}] NHWC pixels: eval {Xe_n.shape}, "
              f"train {Xtr_n.shape}, clean {Xcs_n.shape}, val {Xv_n.shape}")

        # Pixel VAE: conv encoder at latent_dim (cached, content codes).
        accel = Accelerator()
        device = str(accel.device)
        vae = get_or_train_vae(
            dataset=args.dataset, noise_type="symmetric", noise_rate=0.0,
            X_train=np.ascontiguousarray(Xtr_n), config=config,
            device=device, latent_dim=latent_dim)
        if int(getattr(vae, "latent_dim", 0)) != latent_dim:
            raise ValueError(
                f"pixel VAE latent_dim={vae.latent_dim} != LSNPC {latent_dim}")

        if args.emit_only:
            _b = load_bundle(Path("results/ckpt/lsnpc/image") /
                             _lsnpc_ckpt_name(config, args, trainable, True,
                                              cell_tag), device)
            ls = _b["model"].eval()
            # A bundle whose encoder has gone constant must not be emitted as
            # evidence: every cell it produces is the majority class.
            _encoder_liveness(
                ls, torch.as_tensor(np.asarray(Xe_n[:8]), dtype=torch.float32,
                                    device=device), "emit-only")
            best_val_err_h = None
            fit_history = None
            print(f"[emit-only] loaded bundle (training skipped)")
        else:
            # The stage-1 helper sets these from the VAE before building the
            # trainer; this path builds it directly, so mirror both here
            # (z_vae_dim = the pixel VAE's latent, eta = the semi-supervised
            # mixing probability resolved the same way).
            _vae_local = locals().get("vae")
            config.lsnpc_z_vae_dim = int(getattr(_vae_local, "latent_dim", latent_dim))
            config.lsnpc_eta = _resolve_lsnpc_eta(config, True)
            trainer = LSNPCTrainer(model=None, config=config, device=device,
                                   accel=accel)
            trainer.train(
                vae=vae, X_train=np.ascontiguousarray(Xtr_n),
                y_train_noisy=yntr,
                X_clean_set=np.ascontiguousarray(Xcs_n), y_clean_set=ycs,
                n_classes=n_classes,
                X_val=np.ascontiguousarray(Xv_n), y_val_clean=ycv,
                y_val_noisy=ynv,
                val_noisy_labels_are_real=(args.dataset != "animal10n"),
                use_semi=True,
                # No loss term reads the decoder: keep it out of training.
                phase1_wo_decoder=True)
            ls = trainer.model.eval()
            best_val_err_h = trainer.best_val_error
            fit_history = trainer.history
            print(f"val_err_h (best) = {best_val_err_h}")

            # Persist the trained LSNPC for re-use (Set E / emission mode).
            _save_lsnpc_ckpt(
                config, ls, n_classes,
                x_dim=int(np.asarray(Xtr_n).reshape(len(Xtr_n), -1).shape[1]),
                splits={"eval_idx": eval_idx, "val_idx": val_idx,
                        "cs_idx": cs_idx, "tr_idx": tr_idx},
                args=args, trainable=trainable, pixel=True,
                val_err_h=trainer.best_val_error, cell_tag=cell_tag)

        # z_vae latents for the held-out eval slice (same VAE encoder).
        z_vae_eval = _encode_pixel_latents(vae, np.ascontiguousarray(Xe_n), device)
        bs = int(config.batch_size); M = int(getattr(config, "iw_samples", 5))
        proto = compute_protocol_scores(ls, np.ascontiguousarray(Xe_n), yce, yne,
                                        device, batch_size=bs, M=M, seed=args.seed,
                                        z_vae=z_vae_eval, k_perturb=args.rob_k)
        _fx, _fz = np.ascontiguousarray(Xe_n), z_vae_eval
    else:
        # ── Frozen RN50 features: encode everything once, batched on GPU ──
        enc = RN50FeatureEncoder()
        enc = enc.to(dev).eval()

        @torch.no_grad()
        def featurize(X_np: np.ndarray, bs: int = 256) -> np.ndarray:
            dl = place_loader_local(
                tensor_loader(torch.as_tensor(X_np, dtype=torch.float32),
                              batch_size=bs, shuffle=False),
                dev)
            outs = []
            for (xb,) in dl:
                mu, _ = enc.encode(xb)
                outs.append(mu.float().cpu().numpy())
            if not outs:
                # A zero-row input is the unsupervised case (clean_set_size = 0).
                # There is nothing to encode, and np.concatenate([]) raises
                # "need at least one array to concatenate", so probe the
                # encoder once for the feature width and return a correctly
                # shaped (0, D) array. Downstream the trainer's clean-set
                # loader simply yields no batches, which zeroes that term.
                probe = torch.zeros((1, *X_np.shape[1:]), dtype=torch.float32,
                                    device=dev)
                width = int(enc.encode(probe)[0].shape[1])
                return np.zeros((0, width), dtype=np.float32)
            return np.concatenate(outs, axis=0).astype(np.float32)

        t0 = time.time()
        Fe, Ftr, Fcs, Fv = (featurize(a) for a in (Xe, Xtr, Xcs, Xv))
        print(f"RN50 features done: eval {Fe.shape}, train {Ftr.shape}, "
              f"clean {Fcs.shape}, val {Fv.shape} ({time.time()-t0:.0f}s)")
        latent_dim = int(Fe.shape[1])

        config = build_config(args.seed, args.epochs, args.beta, args.batch_size,
                              args.clean_set_size, args.eval_slice, latent_dim,
                              args.dataset, n_classes, eta=args.eta,
                              lsnpc_lr=args.lsnpc_lr,
                              hidden_dim=args.hidden_size, n_blocks=args.n_blocks,
                              focal_gamma=args.focal_gamma,
                              focal_alpha=args.focal_alpha,
                              head=args.lsnpc_head,
                              correction_cond=args.correction_cond,
                              shared_yhat_embed=args.shared_yhat_embed,
                              correction_input=args.correction_input)
        config.output_dir = _lsnpc_output_dir(
            config, args, trainable=False, pixel=False, cell_tag=cell_tag)
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"[out] {config.output_dir}")

        accel = Accelerator()
        device = str(accel.device)

        if args.emit_only:
            _b = load_bundle(Path("results/ckpt/lsnpc/image") /
                             _lsnpc_ckpt_name(config, args, False, False,
                                              cell_tag), device)
            ls = _b["model"].eval()
            best_val_err_h = None
            fit_history = None
            print(f"[emit-only] loaded bundle (training skipped)")
        else:
            # The trainer's "VAE" is an IDENTITY over the frozen features: the
            # 2048-d RN50 embedding IS the latent (embedding-space correction,
            # same as text). Pixel encoding already happened once in featurize().
            # The stage-1 helper sets these from the VAE before building the
            # trainer; this path builds it directly, so mirror both here
            # (z_vae_dim = the pixel VAE's latent, eta = the semi-supervised
            # mixing probability resolved the same way).
            _vae_local = locals().get("vae")
            config.lsnpc_z_vae_dim = int(getattr(_vae_local, "latent_dim", latent_dim))
            config.lsnpc_eta = _resolve_lsnpc_eta(config, True)
            trainer = LSNPCTrainer(model=None, config=config, device=device,
                                   accel=accel)
            trainer.train(vae=IdentityFeatureEncoder(), X_train=Ftr,
                          y_train_noisy=yntr, X_clean_set=Fcs, y_clean_set=ycs,
                          n_classes=n_classes, X_val=Fv, y_val_clean=ycv,
                          y_val_noisy=ynv, use_semi=True,
                          val_noisy_labels_are_real=(args.dataset != "animal10n"))
            ls = trainer.model.eval()
            best_val_err_h = trainer.best_val_error
            fit_history = trainer.history
            print(f"val_err_h (best) = {best_val_err_h}")

            # Persist the trained LSNPC for re-use (Set E / emission mode).
            _save_lsnpc_ckpt(
                config, ls, n_classes, x_dim=int(Ftr.shape[1]),
                splits={"eval_idx": eval_idx, "val_idx": val_idx,
                        "cs_idx": cs_idx, "tr_idx": tr_idx},
                args=args, trainable=False, pixel=False,
                val_err_h=trainer.best_val_error, cell_tag=cell_tag)

        # ── Data-level protocol scores on the held-out eval slice ──
        bs = int(config.batch_size); M = int(getattr(config, "iw_samples", 5))
        proto = compute_protocol_scores(ls, Fe, yce, yne, device,
                                        batch_size=bs, M=M, seed=args.seed,
                                        k_perturb=args.rob_k)
        _fx, _fz = Fe, None
    corr = proto["corr"]
    rows = proto["rows"]
    valid_at_f = {k: v for k, v in proto["scores"].items()
                  if k.startswith("valid_at")}
    mis = proto["mis"]
    succ = proto["succ"]

    # Deployed-frame re-emission (EMIT_FRAMES=1, paper sec:deploy): the
    # self-referenced variants of the two label-referenced axes, computed on
    # the same rows and in the same pass as the instrument values above.
    emit_frame_report(
        config.output_dir, ls, _fx, yce, yne, device, batch_size=bs, M=M,
        seed=args.seed, k_perturb=args.rob_k,
        stored_auroc={r[0]: r[1] for r in rows})

    print(f"\n=== image {args.dataset}: mis n={mis.sum()} "
          f"validity_mis={succ[mis].mean():.3f} ({succ[mis].sum()}/{mis.sum()}) ===")
    print(f"{'feature':24s} {'repAUROC':>8s} {'repAP':>7s} {'valid_mean':>11s} "
          f"{'invalid_mean':>13s} {'dmgAUROC':>8s} {'dmgAP':>7s}")
    for name, a, ap, vm, im, da, dap in rows:
        print(f"{name:24s} {a:8.3f} {ap:7.3f} {vm:11.4f} {im:13.4f} "
              f"{da:8.3f} {dap:7.3f}")
    print(f"transitions: {proto['transitions']}")

    # Per-query export (transition figures).
    os.makedirs(config.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(config.output_dir,
                     f"per_query_scores_s{args.seed}_b{args.beta}.npz"),
        transition=(np.where(mis & succ, "WR",
                     np.where(mis & ~succ, "WW",
                     np.where(~mis & ~succ, "RW", "RR")))),
        **{f"score__{k}": np.asarray(v, dtype=np.float64)
           for k, v in proto["scores"].items()
           if k in SCORE_KEYS and v is not None})

    res = {
        "dataset": args.dataset, "noise": args.noise,
        "noise_type": (f"real_{args.noise_type}" if real
                       else args.noise_type),
        "real_view": (args.noise_type if real else None),
        "encoder": (f"{args.backbone}_trainable" if trainable else
                    f"{args.backbone}_frozen" if pixel else "resnet50_frozen"),
        "label_source": "data",
        "beta": args.beta, "seed": args.seed, "eta": args.eta,
        "clean_set_size": args.clean_set_size,
        "img_size": (int(X.shape[-1]) if pixel else None),
        "rob_k": args.rob_k,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "n_train": len(Xtr), "n_val": len(Xv), "n_eval": len(Xe),
        "mis_frac": float(mis.mean()), "validity_mis": float(succ[mis].mean()),
        "val_err_h": best_val_err_h,
        "cf_auroc": {name: rows[i][1] for i, (name, *_r) in enumerate(rows)},
        "cf_ap": {name: rows[i][2] for i, (name, *_r) in enumerate(rows)},
        "damage_auroc": proto["damage_auroc"],
        "damage_ap": proto["damage_ap"],
        "transitions": proto["transitions"],
        "valid_at": {k: float(v.mean()) for k, v in valid_at_f.items()},
        "corr_acc": float((corr == yce).mean()),
        "corr_f1": multiclass_f1(corr, yce),
        "corr_precision": multiclass_precision(corr, yce),
        "corr_recall": multiclass_recall(corr, yce),
        "noisy_acc": float((yne == yce).mean()),
        "noisy_f1": multiclass_f1(yne, yce),
        "noisy_precision": multiclass_precision(yne, yce),
        "noisy_recall": multiclass_recall(yne, yce),
        "total_train_time_s": (fit_history or {}).get("total_train_time_s", 0.0),
    }
    out = Path(f"{config.output_dir}/result.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:400])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
