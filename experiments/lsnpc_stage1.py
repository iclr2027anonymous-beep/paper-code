"""Stage 1: IW-LSNPC correction (paper v5B).

Extracted from the main pipeline orchestrator to keep each concern
in a single, focused module.
"""
from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from models.lsnpc import build_lsnpc
from trainers.lsnpc import LSNPCTrainer, _encode_vae_latents
from utils.batching import place_loader_local, tensor_loader
from utils.provenance import hash_arrays, prepare_checkpoint_dir
from utils.save import atomic_write_json, load_model_state

from .data import (
    _batched_lsnpc_infer,
    _carve_eval_split,
    _make_lsnpc_x_tensor,
)

log = logging.getLogger(__name__)


# ── Classification-correction diagnostics ─────────────────────────────

def _classification_correction_diagnostics(
    y_true: np.ndarray,
    noisy_prob: np.ndarray,
    corrected_prob: np.ndarray,
    top_classes: tuple[int, ...] = (0,),
    n_bins: int = 10,
) -> dict[str, object]:
    """Correction-quality diagnostics on a clean held-out split.

    Validates the noisy/corrected probability shapes, normalises both to
    probability rows, and reports: clean test size, noisy/corrected
    conditioning error and its reduction, ECE, Brier score, and the
    corrected confusion matrix (paper correction-quality evidence).
    Additionally reports the per-class mean shift of the corrected
    conditioning versus the noisy probabilities (v2 extension).

    ``corrected_prob`` is the LSNPC corrected-conditioning vector
    (probability simplex row, NOT a one-hot argmax), matching the
    semantics of the noise transition matrix Q.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    noisy_prob = np.asarray(noisy_prob, dtype=np.float64)
    corrected_prob = np.asarray(corrected_prob, dtype=np.float64)
    if noisy_prob.shape != corrected_prob.shape:
        raise ValueError(
            "noisy and corrected probabilities must have identical shapes")
    if noisy_prob.ndim != 2 or len(noisy_prob) != len(y_true):
        raise ValueError(
            "classification probabilities must have shape (n_samples, n_classes)")
    if noisy_prob.shape[1] < 2:
        raise ValueError("classification diagnostics require at least two classes")
    if np.any((y_true < 0) | (y_true >= noisy_prob.shape[1])):
        raise ValueError("clean labels fall outside the probability class range")

    def _normalise(prob: np.ndarray) -> np.ndarray:
        prob = np.clip(prob, 0.0, None)
        row_sum = prob.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("probability rows must have positive mass")
        return prob / row_sum

    def _ece(prob: np.ndarray) -> float:
        pred = prob.argmax(axis=1)
        confidence = prob.max(axis=1)
        correct = pred == y_true
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        total = max(1, len(y_true))
        value = 0.0
        for idx in range(n_bins):
            if idx == n_bins - 1:
                in_bin = (confidence >= edges[idx]) & (confidence <= edges[idx + 1])
            else:
                in_bin = (confidence >= edges[idx]) & (confidence < edges[idx + 1])
            if np.any(in_bin):
                value += (
                    float(in_bin.sum()) / total
                    * abs(float(correct[in_bin].mean())
                          - float(confidence[in_bin].mean()))
                )
        return float(value)

    # ── v2 per-class shift report (raw inputs, unchanged semantics) ──
    n = len(y_true)
    y_pred_noisy = noisy_prob.argmax(axis=-1)
    report: dict[str, object] = {
        "clf_noisy_accuracy_full_test":
            float((y_pred_noisy == y_true).mean()) if n else float("nan"),
        "clf_noisy_max_conf_full_test":
            float(noisy_prob.max(axis=-1).mean()) if n else float("nan"),
    }
    correction = corrected_prob - noisy_prob
    for k in top_classes:
        mask = y_pred_noisy == k
        nmk = int(mask.sum())
        if nmk == 0:
            report[f"clf_correction_class_{k}_n"] = 0
            report[f"clf_correction_class_{k}_mean_shift"] = 0.0
            continue
        sub = correction[mask]
        report[f"clf_correction_class_{k}_n"] = nmk
        report[f"clf_correction_class_{k}_mean_shift"] = float(sub[:, k].mean())

    # ── full correction-quality metrics (normalised, validated) ──
    noisy_prob = _normalise(noisy_prob)
    corrected_prob = _normalise(corrected_prob)
    n_classes = noisy_prob.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    noisy_pred = noisy_prob.argmax(axis=1)
    corrected_pred = corrected_prob.argmax(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, corrected_pred), 1)

    noisy_error = float((noisy_pred != y_true).mean())
    corrected_error = float((corrected_pred != y_true).mean())
    report.update({
        "diagnostic_test_size": int(len(y_true)),
        "noisy_conditioning_error": noisy_error,
        "corrected_conditioning_error": corrected_error,
        "conditioning_error_reduction": noisy_error - corrected_error,
        "noisy_ece": _ece(noisy_prob),
        "corrected_ece": _ece(corrected_prob),
        "noisy_brier": float(np.square(noisy_prob - one_hot).sum(axis=1).mean()),
        "corrected_brier": float(
            np.square(corrected_prob - one_hot).sum(axis=1).mean()),
        "corrected_confusion_matrix": confusion.tolist(),
    })
    return report


# ── Leakage audit (paper Thm.) ────────────────────────────────────────

def _audit_leakage(X_train, clean_idx, X_test, y_train_clean,
                   y_train_noisy, X_clean_set=None) -> dict:
    """Data-leakage audit for the LSNPC clean_set/train/test split (paper Thm.).

    The Clean-Set Correction Bound (paper Thm.) requires the clean set D_clean
    to be drawn i.i.d. and **disjoint** from the noisy training data used to
    fit the prior P(φ), and disjoint from the test set. It further requires
    that the clean training label y* is used ONLY on the clean set (never as
    conditioning for the noisy-encoder pass on the full train set).

    This function asserts those invariants and returns a report dict. Raises
    ``AssertionError`` on any violation so a mis-wired split fails loudly.

    ``X_clean_set`` may be passed directly (tabular: the validation split is
    the clean set); otherwise it is derived from ``clean_idx`` into ``X_train``.
    """
    X_train = np.asarray(X_train).reshape(len(X_train), -1)
    X_test = np.asarray(X_test).reshape(len(X_test), -1)

    report: dict = {}

    # 1) Clean-set indices are unique (no duplicated clean-set rows).
    if clean_idx is not None:
        clean_idx = np.asarray(clean_idx)
        n_unique = len(np.unique(clean_idx))
        assert n_unique == len(clean_idx), (
            f"clean_idx has duplicates: {len(clean_idx)} idx, {n_unique} unique")
        report["n_clean_set"] = int(len(clean_idx))
        report["clean_idx_unique"] = True
        X_clean_set = X_train[clean_idx]
    else:
        X_clean_set = np.asarray(X_clean_set).reshape(len(X_clean_set), -1)
        report["n_clean_set"] = int(len(X_clean_set))
        report["clean_idx_unique"] = True

    # 2) Clean-set rows content-overlap with the test set (advisory, non-fatal).
    test_rows = {r.tobytes() for r in np.ascontiguousarray(X_test)}
    overlap = sum(1 for r in np.ascontiguousarray(X_clean_set)
                  if r.tobytes() in test_rows)
    report["clean_test_row_content_overlap"] = int(overlap)
    report["clean_test_row_overlap_fraction"] = float(
        overlap / max(len(X_clean_set), 1)
    )
    if overlap > 0:
        log.warning(
            f"  [leakage audit] {overlap}/{len(X_clean_set)} clean_set rows "
            f"have content-identical matches in X_test "
            f"(typical for tabular datasets with duplicates; NOT label leakage "
            f"because clean_set and test come from a single train/test split)."
        )

    # 3) Clean vs noisy training labels differ (noise was actually injected)
    y_clean = np.asarray(y_train_clean)
    y_noisy = np.asarray(y_train_noisy)
    assert len(y_clean) == len(y_noisy)

    frac_diff = float((y_clean != y_noisy).mean())
    report["train_label_noise_frac"] = frac_diff
    if frac_diff == 0.0:
        log.warning(
            "  [leakage audit] y_train_noisy == y_train_clean "
            "(observed noise 0%%) — verify noise injection is active.")

    report["leakage_ok"] = True
    return report


# ── Stage 1: IW-LSNPC correction ──────────────────────────────────────

def _resolve_lsnpc_eta(config, use_semi: bool) -> float:
    """Effective LSNPC gating weight, honouring an explicit ``--lsnpc-eta``.

    A ``None`` ``lsnpc_eta`` (the CLI default) means "not supplied": resolve to
    the mode default (0.1 for the semi-supervised pass, 0.0 for unsupervised).
    Any explicit value is returned untouched so an eta ablation is a real
    ablation instead of being collapsed onto the default.
    """
    if config.lsnpc_eta is None:
        return 0.1 if use_semi else 0.0
    return float(config.lsnpc_eta)


def resolve_clean_rows(clean_set_size: int, n_available: int) -> int:
    """Number of clean rows to draw from ``n_available`` available rows.

    ``clean_set_size`` semantics (single definition, every consumer):
      -1  use all available clean rows;
       0  no clean set at all (unsupervised correction);
      >0  cap at that many rows, or fewer if the pool is smaller.
    """
    cs = int(clean_set_size)
    if cs < -1:
        raise ValueError(
            f"clean_set_size must be -1 (use all), 0 (no clean set) or a "
            f"positive cap, got {cs}")
    if cs == 0:
        return 0
    if cs < 0:
        return int(n_available)
    return min(cs, int(n_available))


def _select_clean_set(config, rng, val_clean=None, X_train=None,
                      y_train_clean=None, pseudo_clean=None):
    """Build the clean set D_clean (paper Thm) and its index mapping.

    ``config.clean_set_size`` semantics (see ``resolve_clean_rows``):
    -1 = use ALL available clean rows (default); 0 = no clean set at all;
    > 0 = explicit cap.

    Returns ``(X_clean_set, y_clean, clean_idx, m)``:
      - X_clean_set / y_clean: the clean-labelled rows and their labels.
      - clean_idx: indices into X_train when the clean set is drawn from the
        training split; None when it is a separate held-out split
        (val_clean / pseudo_clean).
      - m: the actual number of clean rows selected.

    Training-drawn path (``val_clean is None`` and ``pseudo_clean is None``):
    the clean set is a sub-sample of X_train and the eval split is carved
    from the remainder, so "use all" would consume every training row and
    leave nothing for eval (leakage: the eval split must stay disjoint and
    clean-labelled for checkpoint selection). ``clean_set_size = -1`` (use
    all) is therefore INVALID here and raises a clear error; 0 (no clean set)
    and a positive cap are both fine.
    """
    clean_set_size = int(config.clean_set_size)
    if clean_set_size < -1:
        raise ValueError(
            f"clean_set_size must be -1 (use all), 0 (no clean set) or a "
            f"positive cap, got {clean_set_size}")
    if val_clean is not None:
        # The validation split IS the clean set. It is a proper held-out
        # split (clean labels from the noise-free load), disjoint from both
        # the noisy training set and the test set. Keep 4D image tensors
        # intact — flattening them here crashed the conv-VAE encode with
        # "Expected 3D or 4D input to conv2d, got [256, 3072]" (fixed
        # implementation). Only flatten genuinely non-image (<=3D) features.
        X_feats = np.asarray(val_clean.features.cpu().numpy())
        X_clean_np = (X_feats.reshape(len(val_clean), -1)
                      if X_feats.ndim <= 3 else X_feats)
        y_clean = np.asarray(
            val_clean.targets.cpu().numpy()).ravel()
        n_take = resolve_clean_rows(clean_set_size, len(X_clean_np))
        if n_take == 0:
            # 0 = no clean set: the corrector trains with no clean
            # supervision at all (the unsupervised setting).
            X_clean_set = X_clean_np[:0]
            y_clean = y_clean[:0]
        elif n_take < len(X_clean_np):
            sub = rng.choice(len(X_clean_np), size=n_take, replace=False)
            X_clean_set = X_clean_np[sub]
            y_clean = y_clean[sub]
        else:
            X_clean_set = X_clean_np
        return X_clean_set, y_clean, None, len(X_clean_set)
    if pseudo_clean is not None:
        # Data mode: the full pseudo-clean pool IS the clean set. The
        # pseudo-clean cap (default 0 = use all) keeps heavy encoders
        # (e.g. Swin) within a feasible runtime.
        X_clean_set = np.asarray(pseudo_clean[0])
        y_clean = np.asarray(pseudo_clean[1]).ravel()
        _cap = int(getattr(config, "pseudo_clean_cap", 0))
        if _cap > 0 and len(X_clean_set) > _cap:
            sub = rng.choice(len(X_clean_set), size=_cap, replace=False)
            X_clean_set = X_clean_set[sub]
            y_clean = y_clean[sub]
        return X_clean_set, y_clean, None, len(X_clean_set)
    # Training-drawn path: no held-out clean split exists.
    if clean_set_size < 0:
        raise ValueError(
            "clean_set_size = -1 (use all) is invalid when no held-out clean "
            "validation split is provided: the clean set would be drawn from "
            "the training split, consuming every row and leaving nothing for "
            "the eval split. Pass 0 (no clean set) or a positive cap.")
    if clean_set_size == 0:
        X_clean_set = np.asarray(X_train)[:0]
        y_clean = np.asarray(y_train_clean)[:0]
        return X_clean_set, y_clean, np.array([], dtype=int), 0
    m = min(clean_set_size, len(X_train))
    clean_idx = rng.choice(len(X_train), size=m, replace=False)
    X_clean_set = X_train[clean_idx]
    y_clean = y_train_clean[clean_idx]
    return X_clean_set, y_clean, clean_idx, m


def run_lsnpc_stage1(vae, X_train, y_train_noisy, X_test, n_test,
                      clf, c_soft_train, c_soft_test, y_train_clean,
                      config, device, accel, ckpt_dir=None,
                      y_test_clean=None, val_clean=None,
                      y_test_noisy=None, yhat_source="clf",
                      pseudo_clean=None,
                      phase1_wo_decoder=False):
    """v5B Stage 1: train the IW-LSNPC correction path and produce the
    corrected latent z and corrected conditioning h̃(x, ŷ).

    Returns ``(z_train, z_test, c_soft_train_corr, c_soft_test_corr, lsnpc,
    clean_set_01_loss, diagnostics)``:
      - z_train / z_test: corrected latents z_0 (torch tensors on ``device``)
        used as the corrected latents, replacing the VAE-encoded latents.
      - c_soft_*_corr: corrected soft predictions h̃ = softmax(g_φ(x, z_0)).
        Replaces h(x) as the conditioning signal.
      - lsnpc: the trained module (kept for optional plausibility scoring).
      - clean_set_01_loss: empirical clean_set 0/1 loss (float; NaN if unsupervised).
      - diagnostics: held-out correction and split-integrity measurements.

    For the 4-system ablation matrix, the caller can choose which of the
    two corrections to apply (latent vs condition vs both) via the
    ``use_corrected_latent_only`` and ``use_corrected_condition_only``
    flags in ``config``.
    """
    X_test_q = X_test[:n_test]
    # Classification correction: ŷ = h(x) scalar labels; the module trains
    # with cross-entropy on the correction objective.
    n_classes = int(c_soft_train.shape[1])

    # ── Clean set D_clean: clean-labelled rows (paper Thm) ──
    use_semi = config.use_semi
    val_idx = None  # defined only when a semi eval split is carved
    if use_semi:
        if val_clean is not None:
            # The validation split IS the clean set: a proper held-out split
            # (clean labels from the noise-free load), disjoint from both the
            # noisy training set and the test set.
            rng = np.random.default_rng(config.seed)
            X_clean_set, y_clean, clean_idx, m = _select_clean_set(
                config, rng, val_clean=val_clean)
            clean_cap_note = (
                f"capped at {m}" if m < len(val_clean)
                else "all clean rows used")
            # Train on the full training set; carve an eval split for
            # checkpoint selection. Its clean labels are used for SELECTION
            # only (val_err_h); the rows are excluded from training, so the
            # clean-labelled rows stay exactly the m clean_set rows.
            remaining_idx = np.arange(len(X_train))
            train_idx, val_idx, X_val, y_val_noisy_raw, y_val_clean_raw = (
                _carve_eval_split(
                    rng, X_train, y_train_noisy, y_train_clean, remaining_idx))
            if val_idx is not None:
                log.info(
                    f"LSNPC semi-supervised: {len(train_idx)} train, "
                    f"{len(val_idx)} eval, clean_set = validation split "
                    f"({len(X_clean_set)} clean rows, {clean_cap_note})")
            else:
                log.info(
                    f"LSNPC semi-supervised: no eval split, "
                    f"clean_set = validation split "
                    f"({len(X_clean_set)} clean rows, {clean_cap_note})")
        else:
            # No clean validation split: draw the clean set from the training
            # split, or in data mode use ALL provided clean data (the full
            # test set). An eval split is carved from the remainder for
            # checkpoint selection only.
            if pseudo_clean is not None:
                rng = np.random.default_rng(config.seed)
                X_clean_set, y_clean, clean_idx, m = _select_clean_set(
                    config, rng, pseudo_clean=pseudo_clean)
                remaining_idx = np.arange(len(X_train))
                # Carve independently of the pseudo-clean cap draw (as before).
                rng = np.random.default_rng(config.seed)
                log.info(
                    f"LSNPC semi-supervised: using {m} clean rows as the "
                    f"clean_set (data-mode clean supervision)")
            else:
                rng = np.random.default_rng(config.seed)
                X_clean_set, y_clean, clean_idx, m = _select_clean_set(
                    config, rng, X_train=X_train,
                    y_train_clean=y_train_clean)
                remaining_idx = np.setdiff1d(np.arange(len(X_train)), clean_idx)
            train_idx, val_idx, X_val, y_val_noisy_raw, y_val_clean_raw = (
                _carve_eval_split(
                    rng, X_train, y_train_noisy, y_train_clean, remaining_idx))
            if val_idx is not None:
                log.info(
                    f"LSNPC semi-supervised: {len(train_idx)} train, "
                    f"{len(val_idx)} eval, {m} clean_set")
            else:
                log.info(
                    f"LSNPC semi-supervised: no eval split (n={len(X_train)} < 500), "
                    f"{m} clean_set")
    else:
        # Unsupervised only: no eval, no clean_set.
        X_val, y_val_noisy_raw, y_val_clean_raw = None, None, None
        X_clean_set, y_clean = None, None
        clean_idx = None
        train_idx = np.arange(len(X_train))
        log.info("LSNPC unsupervised only (--use-semi=False): no clean_set, no eval split")

    # ── Split enforcement + data-leakage audit (paper Thm. assumptions) ──
    if use_semi:
        leakage_report = _audit_leakage(
            X_train, clean_idx, X_test, y_train_clean, y_train_noisy,
            X_clean_set=X_clean_set)
        log.info(f"Stage 1 leakage audit: {leakage_report}")

    # Noisy supervision ŷ = h(x): scalar classifier labels.
    if yhat_source == "data":
        # Correct the raw noisy labels directly: no pretrained classifier.
        if use_semi:
            # With pseudo_clean (full test set as clean semi-sup set),
            # clean_idx is None and the clean_set's yhat is its clean label.
            config._clean_set_yhat = np.asarray(
                y_clean if clean_idx is None
                else y_train_noisy[clean_idx]).astype(np.int64)
        yhat_train = np.asarray(y_train_noisy).astype(np.int64)
    else:
        if use_semi:
            config._clean_set_yhat = clf.predict(X_clean_set)
        yhat_train = clf.predict(X_train)

    config.lsnpc_eta = _resolve_lsnpc_eta(config, use_semi)
    # Infer z_vae_dim from the VAE model itself, not from config.
    z_vae_dim = vae.latent_dim
    config.lsnpc_z_vae_dim = z_vae_dim
    trainer = LSNPCTrainer(model=None, config=config, device=device, accel=accel)

    # Checkpoint: skip retraining if loadable and --force-retrain not set.
    if ckpt_dir is None:
        ckpt_dir = prepare_checkpoint_dir(config.output_dir, config)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lsnpc_ckpt = ckpt_dir / "lsnpc.pt"
    lsnpc_history_path = ckpt_dir / "lsnpc_history.json"
    lsnpc_history: dict[str, list] | None = None
    if lsnpc_ckpt.exists() and not getattr(config, "force_retrain", False):
        log.info(f"  [LSNPC] loading cached checkpoint from {lsnpc_ckpt}")
        # Match the trainer's model-building params (reads from config).
        lsnpc = build_lsnpc(
            config,
            x_dim=X_train.reshape(len(X_train), -1).shape[1],
            n_classes=n_classes,
            device=device,
        )
        load_model_state(lsnpc_ckpt, lsnpc, device=device)
        if lsnpc_history_path.exists():
            lsnpc_history = json.loads(lsnpc_history_path.read_text())
        elif getattr(config, "formal_run", False):
            raise RuntimeError(
                f"formal LSNPC checkpoint lacks validation history: "
                f"{lsnpc_history_path}")
    else:
        log.info("  [LSNPC] training from scratch...")
        trainer.train(
            vae=vae, X_train=X_train[train_idx],
            y_train_noisy=yhat_train[train_idx],
            X_clean_set=X_clean_set, y_clean_set=y_clean,
            n_classes=n_classes,
            X_val=X_val, y_val_noisy=y_val_noisy_raw,
            y_val_clean=y_val_clean_raw,
            use_semi=use_semi,
            ckpt_path=str(lsnpc_ckpt),
            phase1_wo_decoder=phase1_wo_decoder,
        )
        lsnpc = trainer.model
        lsnpc_history = trainer.history
        atomic_write_json(lsnpc_history_path, lsnpc_history)

    M = int(config.iw_infer_samples)
    batch_size = int(config.batch_size)

    # Deterministic VAE latent means feed the noisy encoder.
    z_vae_train = _encode_vae_latents(vae, X_train, device, batch_size,
                                      accel=accel)
    z_vae_test_full = _encode_vae_latents(vae, X_test, device, batch_size,
                                          accel=accel)

    lsnpc_img = config.lsnpc_image_data
    X_train_t = _make_lsnpc_x_tensor(X_train, lsnpc_img)
    X_test_t = _make_lsnpc_x_tensor(X_test, lsnpc_img)
    yhat_dtype = torch.long
    yhat_train_t = torch.as_tensor(yhat_train, dtype=yhat_dtype)
    if yhat_source == "data":
        yhat_test = np.asarray(y_test_noisy).astype(np.int64)
    else:
        yhat_test = clf.predict(X_test)
    yhat_test_t = torch.as_tensor(yhat_test, dtype=yhat_dtype)

    lsnpc.eval()
    # Step 1 + 2: batched inference — corrected latent z_0 + corrected conditioning.
    z_train, c_train_corr = _batched_lsnpc_infer(
        lsnpc, X_train_t, z_vae_train, yhat_train_t, batch_size, device, M,
        accel=accel)
    z_test_full, c_test_corr_full = _batched_lsnpc_infer(
        lsnpc, X_test_t, z_vae_test_full, yhat_test_t, batch_size, device, M,
        accel=accel)
    c_train_corr = c_train_corr.cpu().numpy()
    c_test_corr_full = c_test_corr_full.cpu().numpy()
    z_test = z_test_full[:len(X_test_q)]
    c_test_corr = c_test_corr_full[:len(X_test_q)]

    diagnostics: dict[str, object] = {
        "clean_set_size": int(len(X_clean_set)) if X_clean_set is not None else 0,
        "lsnpc_model_train_size": int(len(train_idx)),
        "lsnpc_validation_size": int(len(X_val)) if X_val is not None else 0,
        "corrected_prediction_hash": hash_arrays([
            ("corrected_conditioning_full_test", c_test_corr_full),
        ]),
    }
    if lsnpc_history:
        val_err = np.asarray(lsnpc_history.get("val_err_h", []), dtype=float)
        finite_error = np.isfinite(val_err)
        if finite_error.any():
            best_index = int(
                np.flatnonzero(finite_error)[np.argmin(val_err[finite_error])])
            diagnostics.update({
                "lsnpc_checkpoint_metric": "val_corrected_error",
                "lsnpc_best_epoch": int(lsnpc_history["epoch"][best_index]),
                "lsnpc_best_val_error": float(val_err[best_index]),
                "lsnpc_val_noisy_error_at_best": float(
                    lsnpc_history["val_err_noisy"][best_index]),
            })
    if y_test_clean is not None and clf is not None:
        noisy_prob = np.asarray(clf.predict_proba(X_test))
        diagnostics.update(_classification_correction_diagnostics(
            y_test_clean, noisy_prob, c_test_corr_full))

    # ── Empirical clean_set loss R̂_clean(Q) (paper Thm.) ──
    if use_semi:
        # X/yhat stay on CPU; `place_loader` below moves the batches.
        X_clean_t = _make_lsnpc_x_tensor(X_clean_set, lsnpc_img)
        z_vae_clean = _encode_vae_latents(
            vae, X_clean_set, device, batch_size, accel=accel)
        _clean_set_yhat = getattr(config, "_clean_set_yhat", None)
        if _clean_set_yhat is None:
            _clean_set_yhat = (
                np.asarray(y_train_noisy[clean_idx]).astype(np.int64)
                if yhat_source == "data" else clf.predict(X_clean_set))
        yhat_clean_t = torch.as_tensor(
            np.asarray(_clean_set_yhat), dtype=yhat_dtype)
        amp = (accel.autocast() if accel is not None else nullcontext())
        # Batched so a large clean set cannot OOM in one forward pass.
        _bs = max(1, int(getattr(config, "batch_size", 256)))
        clean_dl = place_loader_local(
            tensor_loader(X_clean_t, z_vae_clean, yhat_clean_t,
                          batch_size=_bs, shuffle=False),
            device)
        clean_corr_parts: list[torch.Tensor] = []
        with amp:
            for xb, zvb, yhb in clean_dl:
                _z_a = lsnpc.sample_corrected_latent(xb, zvb, yhb, M=M)
                _c_a = lsnpc.corrected_conditioning(xb, _z_a)
                clean_corr_parts.append(_c_a.float().cpu())
        c_clean_corr = torch.cat(clean_corr_parts, dim=0)
        pred_clean = c_clean_corr.argmax(dim=-1).numpy()
        clean_set_01_loss = float((pred_clean != y_clean).mean())
        log.info(
            f"Stage 1 done: empirical clean_set loss = {clean_set_01_loss:.4f}")
    else:
        clean_set_01_loss = float("nan")
        log.info("Stage 1 done (unsupervised only, clean_set loss = N/A)")

    if lsnpc_history:
        # Full per-epoch training record — same keys as the text pipeline's
        # result json — so img/tab runs carry the same computation record.
        _et = [t for t in lsnpc_history.get("epoch_time_s", [])
               if isinstance(t, (int, float)) and t == t]
        history_block: dict[str, object] = {
            k: list(v) for k, v in lsnpc_history.items()
            if k in ("epoch", "loss", "recon", "kl_zhat", "kl_z",
                     "clean_set_loss", "val_err_h", "val_err_noisy",
                     "epoch_time_s")
        }
        history_block["total_train_time_s"] = (
            float(sum(_et)) if _et else float("nan"))
        diagnostics["history"] = history_block

    return (
        z_train.detach(),
        z_test.detach(),
        c_train_corr,
        c_test_corr,
        lsnpc,
        clean_set_01_loss,
        diagnostics,
    )
