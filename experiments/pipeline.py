"""Full experiment pipeline: load data -> train -> run methods -> save results."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import accuracy_score

from data_process.base import get_dataset
from experiments.protocol import (
    SCORE_KEYS,
    compute_protocol_scores,
    multiclass_f1,
    multiclass_precision,
    multiclass_recall,
)
from trainers.conv_classifier import ConvClassifier, train_conv_classifier
from trainers.lsnpc import _encode_vae_latents
from trainers.plausibility_vae import get_or_train as get_or_train_universal_plaus_vae
from utils.conditioning import build_predictive_conditioning
from utils.provenance import (
    checkpoint_file_hashes,
    config_hash,
    finalise_checkpoint_index,
    git_commit,
    git_is_dirty,
    hash_arrays,
    hash_file,
    prepare_checkpoint_dir,
    scientific_config,
    utc_now,
)
from utils.save import _to_jsonable, atomic_write_json, load_model_state, save_csv, save_json

from .arguments import ExperimentConfig
from .data import _detect_image_dims, _make_lsnpc_x_tensor, _select_device
from .lsnpc_stage1 import run_lsnpc_stage1

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Single-purpose module-level helpers (extracted from the orchestrators).
# ════════════════════════════════════════════════════════════════════════


def _fmt_metric(v) -> str:
    """Format a metric value for log lines (numbers -> 3-decimal strings)."""
    if isinstance(v, (int, float, np.floating, np.integer)):
        v = f"{v:.3f}"
    return str(v)


def _build_run_id(config: ExperimentConfig, timestamp: str) -> str:
    """Construct the unique run identifier used for output files."""
    return (
        f"{config.dataset}_{config.noise_type}{int(config.noise * 100)}_"
        f"{config.system_name()}_seed{config.seed}_"
        f"{config_hash(config)}_gaussian_{timestamp}"
    )


def _clean_config_for_save(config: ExperimentConfig) -> dict:
    """Strip non-serialisable / private fields from a config for JSON dump."""
    return {
        k: _to_jsonable(v)
        for k, v in config.__dict__.items()
        if not callable(v)
        and k not in ("clf", "x_train_tensor")
        and not k.startswith("_")
    }


def _estimator_class(clf) -> str:
    """Actual estimator class name, for provenance (guard vs. reported config)."""
    return type(clf).__name__


def _save_results(
    out_dir: Path,
    run_id: str,
    config: ExperimentConfig,
    results: dict,
    artifacts: dict,
    timestamp: str,
    provenance: dict,
) -> tuple[Path, Path]:
    """Persist results + cleaned config to JSON and CSV."""
    json_path = save_json(
        out_dir,
        run_id,
        {**_clean_config_for_save(config), "timestamp": timestamp},
        results,
        artifacts=artifacts or None,
        provenance=provenance,
    )
    csv_path = save_csv(
        out_dir,
        run_id,
        results,
        {
            "dataset": config.dataset,
            "noise_type": config.noise_type,
            "noise": config.noise,
            "n_test": config.n_test,
            "timestamp": timestamp,
        },
    )
    return json_path, csv_path


def _eps_cond(clf, X_test, y_test_clean) -> float:
    """Noisy classifier error rate on test labels (= conditioning floor).

    Returns NaN for classifiers whose scoring protocol does not apply
    (e.g. structured cost predictors) rather than raising.
    """
    try:
        if hasattr(clf, "score"):
            return 1.0 - clf.score(X_test, y_test_clean)
        if hasattr(clf, "predict"):
            return 1.0 - accuracy_score(y_test_clean, clf.predict(X_test))
    except (ValueError, TypeError):
        # A scorer that exists but raises is a bug (shape/label mismatch),
        # not an inapplicable protocol — surface it, don't blend it into
        # the legitimate NaN path.
        log.warning("eps_cond scoring failed; reporting NaN", exc_info=True)
        return float("nan")
    return float("nan")


def _print_summary(results: dict) -> None:
    """Log a compact per-method metrics table at the end of a run."""
    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    log.info(
        f"  {'Method':<15s} | {'Gain':>10s} | {'Plaus':>10s} | "
        f"{'Valid':>8s} | {'VClean':>8s} | {'Time':>8s}"
    )
    log.info("  " + "-" * 78)
    for name, m in results.items():
        if "error" in m:
            log.info(
                f"  {name:<15s} | {'ERROR':>10s} | {'':>10s} | {'':>8s} | {'':>8s} |"
            )
            continue
        log.info(
            f"  {name:<15s} | {_fmt_metric(m.get('gain')):>10s} | {_fmt_metric(m.get('plausibility')):>10s} | "
            f"{_fmt_metric(m.get('validity')):>8s} | "
            f"{_fmt_metric(m.get('valid_clean')):>8s} | "
            f"{_fmt_metric(m.get('runtime_ms_per_query')):>7s}ms"
        )


# ════════════════════════════════════════════════════════════════════════
# Full experiment orchestration
# ════════════════════════════════════════════════════════════════════════


def _build_classifier(config, device, data_ckpt_dir, base_tag,
                      X_train_clean, y_train_noisy, y_train_clean,
                      y_test_clean, X_test, val):
    """Train or load the downstream ConvClassifier.

    Every loader in the tree returns ``feature_types = None``, so this is the
    only classifier path; the sklearn ``train_classifiers`` alternative was
    unreachable and has been removed.
    """
    img_h, img_w, img_c = _detect_image_dims(X_train_clean)
    # Class count from the data (CIFAR-10N has 10; binary sets have 2).
    n_classes_data = int(max(y_train_clean.max(), y_test_clean.max(),
                             y_train_noisy.max())) + 1

    clf_ckpt = data_ckpt_dir / f"convclf_{base_tag}.pt"
    if clf_ckpt.exists() and not config.force_retrain:
        log.info(f"Loading ConvClassifier from {clf_ckpt}...")
        clf = ConvClassifier(img_channels=img_c, img_size=img_h,
                             n_classes=n_classes_data)
        load_model_state(clf_ckpt, clf)
    else:
        log.info("Training ConvClassifier...")
        clf = train_conv_classifier(
            X_train_clean,
            y_train_noisy,
            val.features.cpu().numpy(),
            val.targets.cpu().numpy(),
            img_channels=img_c,
            img_size=img_h,
            n_classes=n_classes_data,
            device=device,
        )
        torch.save(clf.state_dict(), clf_ckpt)
        log.info(f"Saved ConvClassifier to {clf_ckpt}")
    log.info(f"ConvClassifier: test acc={clf.score(X_test, y_test_clean):.3f}")
    return clf


def _build_conditioning(config, clf, X_train_clean, X_test, n_test):
    """Predictive-posterior conditioning for the train and test splits."""
    c_soft_train = build_predictive_conditioning(
        clf, X_train_clean, mode=config.conditioning_mode
    )
    c_soft_test = build_predictive_conditioning(
        clf, X_test[:n_test], mode=config.conditioning_mode
    )
    return c_soft_train, c_soft_test


def _build_accelerator():
    """Create the Accelerator; return it with the resolved device.

    Accelerator is the default path even on one GPU (no DDP overhead) and
    gives mixed precision, device placement, and multi-GPU readiness. If it
    fails to initialise we raise rather than fall back to raw PyTorch — a
    silent fallback changes numerics and hides real failures.
    Mixed precision: bf16 > fp16 > no, by CUDA capability.
    """
    if torch.cuda.is_available():
        bf16_ok = (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        mp = "bf16" if bf16_ok else "fp16"
    else:
        mp = "no"
    accel = Accelerator(mixed_precision=mp)
    device = accel.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    log.info(f"Using Hugging Face Accelerator (device={device}, mp={mp})")
    return accel, device


def run_experiment(config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """Run a full LSNPC label-correction experiment from config.

    This is the main orchestration function: loads data, trains classifiers,
    runs selected methods, computes metrics, and saves results.
    """
    config.validate()
    repo_root = Path(__file__).resolve().parents[2]
    repo_dirty = git_is_dirty(repo_root)
    if config.formal_run and repo_dirty is not False:
        raise RuntimeError(
            "formal runs require a clean Git worktree so every result maps "
            "to one committed source state"
        )
    started_at = utc_now()
    run_config_hash = config_hash(config)
    config_snapshot = scientific_config(config)
    log.info(f"Config hash: {run_config_hash}")
    log.info(f"Config: {config.__dict__}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _build_run_id(config, timestamp)

    # ── Load data ──
    # The clean split (noise_rate=0.0) supplies y_train_clean for
    # `valid_clean`; the noisy split comes from the same loader with the
    # requested noise rate (injection happens inside `get_dataset`).
    # config.seed is passed as random_state so the split AND the noise vary
    # across seeds — otherwise the multi-seed protocol degenerates (§10.2).
    rs = config.seed
    train_clean, val, test, _ft, _ = get_dataset(
        config.dataset,
        config.data_dir,
        noise_rate=0.0,
        random_state=rs,
    )
    train_noisy, val_noisy, test_noisy, _, _ = get_dataset(
        config.dataset,
        config.data_dir,
        noise_rate=config.noise,
        noise_type=config.noise_type,
        random_state=rs,
    )
    X_train_clean = train_clean.features.cpu().numpy()
    y_train_clean = train_clean.targets.cpu().numpy()
    y_train_noisy = train_noisy.targets.cpu().numpy()  # noisified at load
    # Use noisy test split so evaluation matches training distribution
    # (noise is in the cost grids, not the images themselves).
    X_test = test_noisy.features.cpu().numpy()
    y_test_noisy = test_noisy.targets.cpu().numpy()
    y_test_clean = test.targets.cpu().numpy()  # clean targets for valid_clean
    n_test = len(X_test) if config.n_test <= 0 else min(config.n_test, len(X_test))
    log.info(
        f"Data: {X_train_clean.shape[0]} train, {X_test.shape[0]} test, {X_test.shape[1]} dims"
    )

    base_tag = f"{config.dataset}_{config.noise_type}{int(config.noise * 100)}"
    ckpt_dir = prepare_checkpoint_dir(out_dir, config)
    log.info(f"Isolated checkpoint directory: {ckpt_dir}")
    # Data-level model cache (classifiers / universal VAE), keyed by
    # dataset + noise + seed so config changes reuse them; the scientific
    # component (LSNPC) stays in ckpt_dir.
    data_ckpt_dir = out_dir / "ckpt" / f"{base_tag}_seed{rs}"
    data_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Data-level checkpoint directory: {data_ckpt_dir}")

    data_split_hash = hash_arrays(
        [
            ("X_train_clean", X_train_clean),
            ("y_train_clean", y_train_clean),
            ("y_train_noisy", y_train_noisy),
            ("X_test", X_test),
            ("y_test_clean", y_test_clean),
            ("y_test_noisy", y_test_noisy),
        ]
    )
    observed_noise_rate = float(
        np.not_equal(
            np.asarray(y_train_clean),
            np.asarray(y_train_noisy),
        ).mean()
    )

    # ── Get device (CUDA → CPU) — needed for classifier training ──
    device = _select_device()
    log.info(f"Device: {device}")

    # ── Train the classifier ──
    clf = _build_classifier(
        config, device, data_ckpt_dir, base_tag,
        X_train_clean, y_train_noisy, y_train_clean, y_test_clean, X_test, val,
    )

    # ── Build predictive-posterior conditioning ──
    c_soft_train, c_soft_test = _build_conditioning(
        config, clf, X_train_clean, X_test, n_test,
    )

    # ── Create Accelerator (always-on default) ──
    accel, device = _build_accelerator()

    # ── Train/load universal plausibility VAE for VAE-plausibility ──
    # Wrapper handles caching keyed by (dataset, noise_type, noise_rate); if
    # the checkpoint is missing, it auto-trains and saves before returning.
    universal_vae = get_or_train_universal_plaus_vae(
        dataset=config.dataset,
        noise_type=config.noise_type,
        noise_rate=config.noise,
        X_train=X_train_clean,
        config=config,
        device=device,
        accel=accel,
        ckpt_dir=data_ckpt_dir,
        force_retrain=config.force_retrain,
        latent_dim=config.latent_dim,
        use_augment=bool(getattr(config, "vae_augment", True)),
    )

    # Prepare the VAE with the accelerator so all baseline methods see
    # a correctly device-placed model (not a raw checkpoint instance).
    if accel is not None:
        universal_vae = accel.prepare(universal_vae)
        universal_vae.eval()

    results: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}

    # ── Stage 1: IW-LSNPC label correction (paper §4) ──
    # Systems (config.system): no_lsnpc (S1 baseline) | corrected_latent_only
    # (S2) | corrected_condition_only (S3) | full_lsnpc (S4) | lsnpc_train_only.
    eps_cond = _eps_cond(clf, X_test[:n_test], y_test_clean[:n_test])
    lsnpc_active = config.system != "no_lsnpc"
    if lsnpc_active:
        log.info("Stage 1: training IW-LSNPC correction path...")
        lsnpc_vae = universal_vae
        (_, _, _, _, lsnpc_module, clean_set_01_loss, lsnpc_diagnostics) = (
            run_lsnpc_stage1(
                lsnpc_vae,
                X_train_clean,
                y_train_noisy,
                X_test,
                n_test,
                clf,
                c_soft_train,
                c_soft_test,
                y_train_clean,
                config,
                device,
                accel,
                ckpt_dir=ckpt_dir,
                y_test_clean=y_test_clean,
                val_clean=val,
            )
        )

        # Freeze LSNPC: no gradients flow into its weights after stage 1.
        lsnpc_module.eval()
        for p in lsnpc_module.parameters():
            p.requires_grad_(False)
        log.info("LSNPC frozen (eval mode, requires_grad=False)")

        # ── Protocol scores (all five axes, every system) ──
        # Same per-query scores as the text pipeline, on the eval slice:
        # X = the model's x layout, y_noisy = the labels the correction sees
        # (clf predictions), y_clean = oracle. No try/except — a score
        # failure must crash the run loudly.
        _lsnpc_img = bool(getattr(config, "lsnpc_image_data", False))
        X_eval_np = X_test[:n_test]
        X_eval_x = (_make_lsnpc_x_tensor(X_eval_np, _lsnpc_img)
                    .numpy())
        y_eval_noisy = clf.predict(X_eval_np)
        _zvae = _encode_vae_latents(
            lsnpc_vae, X_eval_np, device,
            int(getattr(config, "batch_size", 256)), accel=accel
        ).float().cpu().numpy()
        proto = compute_protocol_scores(
            lsnpc_module, X_eval_x, y_test_clean[:n_test],
            y_eval_noisy, device,
            batch_size=int(getattr(config, "batch_size", 256)),
            M=int(getattr(config, "iw_infer_samples", 5)),
            seed=int(getattr(config, "seed", 42)),
            z_vae=_zvae)
        proto_auroc = {name: a for name, a, *_ in proto["rows"]}
        proto_ap = {name: ap for name, _, ap, *_ in proto["rows"]}
        proto_auroc["validity_mis"] = float(proto["succ"][proto["mis"]].mean())
        proto_auroc["mis_frac"] = float(proto["mis"].mean())
        proto_auroc["corr_acc"] = float(proto["succ"].mean())
        # Flat corrected/noisy quality on the eval slice vs the clean oracle:
        # corr = corrected labels, noisy = the label stream the correction
        # saw (clf predictions). Recorded on EVERY run (paper rule) — the
        # beta-sweep figure needs corr/noisy F1 + acc for every modality.
        y_eval_clean = y_test_clean[:n_test]
        corr_labels = proto["corr"]
        results_std: dict[str, Any] = {
            "corr_acc": float((corr_labels == y_eval_clean).mean()),
            "corr_f1": multiclass_f1(corr_labels, y_eval_clean),
            "corr_precision": multiclass_precision(corr_labels, y_eval_clean),
            "corr_recall": multiclass_recall(corr_labels, y_eval_clean),
            "noisy_acc": float((y_eval_noisy == y_eval_clean).mean()),
            "noisy_f1": multiclass_f1(y_eval_noisy, y_eval_clean),
            "noisy_precision": multiclass_precision(y_eval_noisy, y_eval_clean),
            "noisy_recall": multiclass_recall(y_eval_noisy, y_eval_clean),
            # Full parity with the text pipeline's flat result fields.
            "flip_frac": float((corr_labels != y_eval_noisy).mean()),
            "mis_frac": float(proto["mis"].mean()),
            "validity_mis": float(proto["succ"][proto["mis"]].mean()),
            "n_eval": int(len(corr_labels)),
        }
        _vat = {k: float(v.mean()) for k, v in proto["scores"].items()
                if k.startswith("valid_at")}
        if _vat:
            results_std["valid_at"] = _vat
        log.info("  protocol scores (repair | damage channels):")
        for name, a, ap, vm, im, da, dap in proto["rows"]:
            log.info(f"    {name:24s} repAUROC={a:.3f} repAP={ap:.3f} "
                     f"dmgAUROC={da:.3f} dmgAP={dap:.3f}")
        log.info(f"  transitions: {proto['transitions']}")

        # Per-query score export for the transition-distribution figure:
        # scores (main axes) + the transition label of each query, saved as
        # a sidecar .npz next to the result json.
        _labels = np.full(len(proto["mis"]), "RR", dtype=object)
        _labels[proto["mis"] & proto["succ"]] = "WR"
        _labels[proto["mis"] & ~proto["succ"]] = "WW"
        _labels[~proto["mis"] & ~proto["succ"]] = "RW"
        _npz_path = out_dir / f"per_query_scores_{config.seed}_{run_config_hash[:8]}.npz"
        np.savez_compressed(
            str(_npz_path),
            allow_pickle=False,
            transition=_labels.astype(str),
            **{f"score__{k}": np.asarray(v, dtype=np.float64)
               for k, v in proto["scores"].items()
               if k in SCORE_KEYS and v is not None})
        log.info(f"  per-query scores written: {_npz_path.name}")

        results["lsnpc"] = {
            "eps_cond": eps_cond,
            "clean_set_01_loss": clean_set_01_loss,
            "tilde_h_clean_agreement": (
                float(1.0 - clean_set_01_loss)
                if clean_set_01_loss == clean_set_01_loss
                else float("nan")
            ),
            **(lsnpc_diagnostics or {}),
            **results_std,
            "cf_auroc": proto_auroc,
            "cf_ap": proto_ap,
            "damage_auroc": proto["damage_auroc"],
            "damage_ap": proto["damage_ap"],
            "transitions": proto["transitions"],
        }
        log.info(
            f"  lsnpc: clean_set_01_loss={_fmt_metric(clean_set_01_loss)}  "
            f"eps_cond={_fmt_metric(eps_cond)}"
        )
    else:
        log.info("LSNPC inactive (baseline system): recording eps_cond only")
        results["baseline"] = {"eps_cond": eps_cond}

    # ── Save results ──
    # Freeze the checkpoint index only when model files exist (input-space-only
    # baseline runs produce none, so provenance is an empty hash set).
    checkpoint_files = checkpoint_file_hashes(ckpt_dir)
    if checkpoint_files:
        finalise_checkpoint_index(ckpt_dir, config)
        checkpoint_hashes = {
            str((ckpt_dir / rel).relative_to(out_dir)): digest
            for rel, digest in checkpoint_files.items()
        }
    else:
        checkpoint_hashes = {}
    provenance = {
        "schema_version": 1,
        "config_hash": run_config_hash,
        "protocol_id": config.protocol_id,
        "formal_run": bool(config.formal_run),
        "system": config.system_name(),
        "configuration_explicit": {
            "system": bool(config.system_explicit),
            "lsnpc_loss": bool(config.lsnpc_loss_explicit),
        },
        "scientific_config": config_snapshot,
        "git_commit": git_commit(repo_root),
        "git_dirty": repo_dirty,
        "data_split_hash": data_split_hash,
        "data": {
            "requested_noise_rate": float(config.noise),
            "observed_noise_rate": observed_noise_rate,
            "n_train": int(len(X_train_clean)),
            "n_val": int(len(val)),
            "n_test": int(len(X_test)),
            "estimator": _estimator_class(clf),
        },
        "checkpoint_hashes": checkpoint_hashes,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    json_path, csv_path = _save_results(
        out_dir, run_id, config, results, artifacts, timestamp, provenance
    )
    provenance["result_hashes"] = {
        json_path.name: hash_file(json_path),
        csv_path.name: hash_file(csv_path),
    }
    manifest_path = out_dir / f"{run_id}.manifest.json"
    atomic_write_json(
        manifest_path,
        {"run_id": run_id, "status": "complete", "provenance": provenance},
    )
    log.info(f"Saved run manifest to {manifest_path}")

    # ── Print summary ──
    _print_summary(results)

    return results
