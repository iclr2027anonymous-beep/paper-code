"""Stage-1 IW-LSNPC label-correction trainer (v5B).

Trains the LSNPC correction path (paper §4 "Stage 1: IW-LSNPC training" +
Appendix C) with the importance-weighted ELBO and a two-phase schedule:

  * Unsupervised pass  — noisy dataset, maximise L_K (Eq. C.6).
  * Semi-supervised pass — clean set, maximise
    L_K^semi + λ·L_clean (Eqs. C.9, 6).

The noisy encoder conditions on the frozen VAE latent
``z_vae = VAE_enc(x)`` (deterministic preprocessing per Appendix C).

``LSNPCTrainer`` extends ``Trainer`` following the VAE pattern:
``train()`` builds everything and delegates to ``train_model()``;
``train_one_epoch()`` runs two phases; ``eval_and_save()`` handles
validation and best-model tracking.

Usage::

    trainer = LSNPCTrainer(model=None, config=config, device=device)
    trainer.train(vae=vae, X_train=X_train, y_train_noisy=yhat_train,
                  X_clean_set=X_clean_set, y_clean_set=y_clean_set,
                  n_classes=n_classes)
    lsnpc, history = trainer.model, trainer.history
"""
from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from models.lsnpc import LSNPC, build_lsnpc
from trainers.base import AccelLike, Trainer
from utils.batching import place_loader_local, tensor_loader
from utils.vae_decode import unbox_decoder_output

log = logging.getLogger(__name__)


@torch.no_grad()
def _encode_vae_latents(vae, X: np.ndarray, device: str,
                        batch_size: int = 256,
                        accel: AccelLike | None = None) -> torch.Tensor:
    """Deterministic VAE latent means z_vae = VAE_enc(x) (μ head only).

    Handles both unconditional (VAE) and conditional VAEs — for a
    conditional VAE we cannot supply c here, so we fall back to the
    unconditional encode signature. The pipeline uses the unconditional VAE
    for the LSNPC path.
    """
    vae.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    dl = place_loader_local(tensor_loader(X_t, batch_size=batch_size), device)
    mus = []
    amp = (accel.autocast() if accel is not None else nullcontext())
    with amp:
        for (xb,) in dl:
            out = vae.encode(xb)
            mu = unbox_decoder_output(out)
            mus.append(mu.detach())
    return torch.cat(mus, dim=0)


@torch.no_grad()
def _macro_f1(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Macro-averaged F1 (per-class F1 mean), robust to missing classes."""
    if pred.numel() == 0:
        return float("nan")
    n_classes = int(max(int(pred.max().item()), int(y.max().item()))) + 1
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum().item())
        fp = int(((pred == c) & (y != c)).sum().item())
        fn = int(((pred != c) & (y == c)).sum().item())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        if prec + rec > 0:
            f1s.append(2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def _encoder_liveness(lsnpc: LSNPC, batch_x: torch.Tensor, where: str,
                      ) -> float | None:
    """Spread of the image encoder's mu over a batch; abort if it is dead.

    Returns ``None`` when the run has no image encoder to check (text runs),
    the mean per-dim standard deviation otherwise. ``spread == 0`` means the
    pooled backbone feature is constant, so the mu head is a constant and
    every downstream accuracy is the majority class no matter what the label
    stream does. That failure is silent and harmless-looking in the logs (it
    produced a full set of degenerate paper rows once), so a dead encoder
    aborts the run rather than being logged and ignored.
    """
    enc = getattr(getattr(lsnpc, "noisy_encoder", None), "image_encoder", None)
    if enc is None or not getattr(lsnpc, "image_data", False):
        return None
    if batch_x.dim() != 4:
        raise ValueError(
            f"[{where}] expected a 4-D image batch, got {tuple(batch_x.shape)}")
    if batch_x.size(0) < 2:
        return None  # std needs more than one row
    # Measured in eval mode: that is how the featuriser is read downstream, and
    # a train-mode forward would mutate the running statistics whenever
    # BatchNorm is not frozen.
    was_training = enc.training
    enc.eval()
    try:
        mu, _ = enc(batch_x)
    finally:
        enc.train(was_training)
    if not bool(torch.isfinite(mu).all()):
        raise RuntimeError(
            f"[{where}] image-encoder latents are non-finite: the run has "
            "diverged. Refusing to continue "
            ".")
    spread = float(mu.detach().float().std(dim=0).mean())
    if spread == 0.0:
        raise RuntimeError(
            f"[{where}] the image encoder is DEAD: its mu is constant across "
            "the batch, so every downstream accuracy is the majority class "
            "regardless of the label stream. Refusing to continue "
            ".")
    return spread


@torch.no_grad()
def _backbone_liveness(lsnpc: LSNPC, batch_x: torch.Tensor, where: str,
                       ) -> float | None:
    """Spread of the pooled backbone feature, *before* the embedding head.

    ``_encoder_liveness`` tracks the encoder's ``mu``, which the mu/logvar
    heads can rescale independently of the stream they read: a collapsed
    residual stream has been observed to come back as a healthy-looking ``mu``
    , so the ``mu`` spread reports "alive" for a backbone whose
    features are noise. The torchvision, swin and vit encoders all expose
    ``backbone_feature``; ``conv`` does not and is not covered here.
    """
    enc = getattr(getattr(lsnpc, "noisy_encoder", None), "image_encoder", None)
    if enc is None or not getattr(lsnpc, "image_data", False):
        return None
    feature = getattr(enc, "backbone_feature", None)
    if feature is None:
        return None
    if batch_x.dim() != 4:
        raise ValueError(
            f"[{where}] expected a 4-D image batch, got {tuple(batch_x.shape)}")
    if batch_x.size(0) < 2:
        return None  # std needs more than one row
    was_training = enc.training
    enc.eval()
    try:
        h = feature(batch_x)
    finally:
        enc.train(was_training)
    if not bool(torch.isfinite(h).all()):
        raise RuntimeError(
            f"[{where}] backbone features are non-finite: the run has "
            "diverged. Refusing to continue "
            ".")
    spread = float(h.detach().float().std(dim=0).mean())
    if spread == 0.0:
        raise RuntimeError(
            f"[{where}] the image backbone is DEAD: its pooled feature is "
            "constant across the batch, so the encoder's embedding carries no "
            "image information. Refusing to continue "
            ".")
    return spread


@torch.no_grad()
def _lsnpc_val_metrics(
    lsnpc: LSNPC, val_dl: DataLoader, K: int,
    accel: AccelLike | None = None,
) -> dict:
    """Compute validation metrics for LSNPC, batched to avoid GPU OOM.

    ``val_dl`` is ``accel.prepare``-d by the caller, so its batches are already
    on the device and nothing here calls ``.to(device)``.

    It yields tuples ``(x, z_vae, y_hat)`` or
    ``(x, z_vae, y_hat, y_clean)`` depending on whether the clean val
    labels are available.

    Returns dict with:
      * ``val_err_noisy`` — error rate of the noisy label ŷ vs clean y*.
      * ``val_err_h`` — error rate of the corrected predictor h̃ vs clean y*.
    ``val_err_*`` are NaN when no clean label is supplied.
    """
    was_training = lsnpc.training
    lsnpc.eval()

    all_h_preds: list[torch.Tensor] = []
    all_y_clean: list[torch.Tensor] = []
    all_y_hat: list[torch.Tensor] = []
    has_clean = False
    # Detect whether the val DL was built with the optional y_clean
    # column by inspecting the first batch's arity.
    first_batch = next(iter(val_dl))
    has_clean = len(first_batch) == 4
    # Rewind: rebuild the iterator so we start from batch 0.
    val_iter = iter(val_dl)

    amp = (accel.autocast() if accel is not None else nullcontext())
    with amp:
        for items in val_iter:
            if has_clean:
                xb, zv, yh, yc = items
                # Batches are already on device via accel.prepare; no .to() needed.
                z0 = lsnpc.sample_corrected_latent(xb, zv, yh, M=K)
                h_logits = lsnpc.corrected_logits(xb, z0)
                all_h_preds.append(h_logits.detach().cpu())
                all_y_clean.append(yc.detach().cpu())
                all_y_hat.append(yh.detach().cpu())

    out = {}
    if has_clean and all_h_preds:
        h_preds = torch.cat(all_h_preds, dim=0)
        yc_all = torch.cat(all_y_clean, dim=0)
        yh_all = torch.cat(all_y_hat, dim=0)
        h_pred = h_preds.argmax(dim=-1)
        out["val_err_h"] = float((h_pred != yc_all).float().mean().item())
        out["val_err_noisy"] = float((yh_all != yc_all).float().mean().item())
        out["val_f1_h"] = _macro_f1(h_pred, yc_all)
        out["val_f1_noisy"] = _macro_f1(yh_all, yc_all)
    else:
        out["val_err_h"] = float("nan")
        out["val_err_noisy"] = float("nan")

    if was_training:
        lsnpc.train()
    return out


class LSNPCTrainer(Trainer):
    """LSNPC trainer following the VAE pattern: train() → train_model() →
    train_one_epoch() + eval_and_save().

    Checkpoints the best model by val_err_h (the corrected-error metric on
    the clean val labels).  Stores per-epoch metrics in ``self.history``.
    """

    @property
    def best_val_error(self) -> float:
        """Best corrected-predictor validation error (val_err_h) over training."""
        return -self._best_score if self._best_score != -float("inf") else float("nan")

    def _param_groups(self, lr: float) -> list[dict]:
        """AdamW param groups, with the image backbone scaled by bb_lr_scale.

        Only a model that owns an image encoder with a ``.net`` backbone
        contributes a second group; everything else falls through to the
        single-group behaviour, so text/embedding runs are untouched.
        """
        scale = float(getattr(self, "_bb_lr_scale", 1.0))
        if scale == 1.0:
            return [{"params": list(self.model.parameters()), "lr": lr}]
        # An LSNPC owns TWO backbones (the posterior encoder's and the label
        # head's) and no ``image_encoder`` attribute of its own, so the
        # backbone group must be collected from both; resolving it on the
        # model alone leaves the group empty and the scale a silent no-op.
        encs = [getattr(self.model, "image_encoder", None),
                getattr(getattr(self.model, "noisy_encoder", None),
                        "image_encoder", None),
                getattr(getattr(self.model, "label_head", None),
                        "image_encoder", None)]
        backbone_ids: set[int] = set()
        for enc in encs:
            if enc is not None and hasattr(enc, "net"):
                backbone_ids |= {id(p) for p in enc.net.parameters()}
        bb, rest = [], []
        for p in self.model.parameters():
            (bb if id(p) in backbone_ids else rest).append(p)
        return [{"params": rest, "lr": lr},
                {"params": bb, "lr": lr * scale}]

    def __init__(self, model=None, config=None, device="cpu", accel=None,
                 mixed_precision=None):
        super().__init__(model, config, device, accel, mixed_precision)

    # ── Public entry point ──────────────────────────────────────────
    def train(
        self,
        vae,
        X_train: np.ndarray,
        y_train_noisy: np.ndarray,
        X_clean_set: np.ndarray | None = None,
        y_clean_set: np.ndarray | None = None,
        n_classes: int = 2,
        X_val: np.ndarray | None = None,
        y_val_clean: np.ndarray | None = None,
        y_val_noisy: np.ndarray | None = None,
        # Whether the validation's noisy labels are a real annotation stream.
        # animal10n supplies clean labels as the encoder input because its
        # clean test split is the only labelled source, so val_err_noisy /
        # val_f1_noisy would compare a placeholder against itself.
        val_noisy_labels_are_real: bool = True,
        ckpt_path: str | None = None,
        use_semi: bool = False,
        phase1_wo_decoder: bool = False,
    ) -> "LSNPCTrainer":
        """Build data pipeline + model + opt + dl, then delegate to
        ``train_model()`` (inherited from ``Trainer``).

        Returns ``self`` so that ``trainer.model`` / ``trainer.history``
        are accessible after training.
        """
        cfg = self.config
        device = self.device
        accel = self.accel

        # ── Hyper-parameters ──────────────────────────────────────
        latent_dim = int(cfg.latent_dim)
        z_vae_dim = int(cfg.lsnpc_z_vae_dim)
        K = int(cfg.iw_samples)
        lr = float(cfg.lsnpc_lr)
        epochs = int(cfg.lsnpc_epochs)
        batch_size = int(cfg.batch_size)
        is_image_data = bool(cfg.lsnpc_image_data)
        x_dim = int(np.asarray(X_train).reshape(len(X_train), -1).shape[1])
        eff_dim = z_vae_dim if is_image_data else x_dim

        log.info(
            f"Stage 1 (LSNPC): x_dim={eff_dim} "
            f"latent_dim={latent_dim} "
            f"K={K} nu=learned(x,yhat) epochs={epochs} lr={lr} "
            f"clean_set={0 if X_clean_set is None else len(X_clean_set)}"
        )

        # ── Build model ────────────────────────────────────────────
        self.model = build_lsnpc(
            cfg,
            x_dim=x_dim,
            n_classes=n_classes,
            device=device,
        )
        is_image_data = self.model.image_data

        # No loss term reads the data decoder, so keep its parameters out of
        # training (no AdamW update, no weight decay).
        self.model.phase1_wo_decoder = bool(phase1_wo_decoder)
        if phase1_wo_decoder and hasattr(self.model, "data_decoder"):
            for _p in self.model.data_decoder.parameters():
                _p.requires_grad_(False)
            log.info("  [LSNPC] Phase 1 WITHOUT decoder: data_decoder frozen "
                     "out")

        # ── Precompute VAE latents + prepare CPU tensors ────────────
        z_vae_train = _encode_vae_latents(vae, X_train, device, batch_size,
                                          accel=self.accel).cpu()
        X_train_t = torch.as_tensor(
            np.asarray(X_train).reshape(len(X_train), -1), dtype=torch.float32)
        X_train_4d = None
        if is_image_data:
            xt = np.asarray(X_train)
            if xt.ndim == 4:
                X_train_4d = torch.as_tensor(
                    xt, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train_t = torch.as_tensor(np.asarray(y_train_noisy), dtype=torch.long)

        # Clean set (semi-supervised)
        has_clean_set = (
            X_clean_set is not None and y_clean_set is not None
            and len(X_clean_set) > 0
        )
        if has_clean_set:
            assert X_clean_set is not None and y_clean_set is not None
            z_vae_clean = _encode_vae_latents(
                vae, X_clean_set, device, batch_size, accel=self.accel).cpu()
            X_clean_t = torch.as_tensor(
                np.asarray(X_clean_set).reshape(len(X_clean_set), -1),
                dtype=torch.float32)
            X_clean_4d = None
            if is_image_data:
                xa = np.asarray(X_clean_set)
                if xa.ndim == 4:
                    X_clean_4d = torch.as_tensor(
                        xa, dtype=torch.float32).permute(0, 3, 1, 2)
            y_clean_t = torch.as_tensor(
                np.asarray(y_clean_set), dtype=torch.long)
            yhat_clean = getattr(cfg, "_clean_set_yhat", None)
            if yhat_clean is not None:
                yhat_clean_t = torch.as_tensor(yhat_clean, dtype=torch.long)
            else:
                yhat_clean_t = y_clean_t.clone()
            # Per-row confidence on the clean supervision: 1.0 for the rows
            # that are known clean, <1.0 for rows whose label is itself a
            # correction (Set-F admission weight). Checked for length so a
            # misaligned vector fails here instead of weighting wrong rows.
            _cw = getattr(cfg, "_clean_set_weight", None)
            if _cw is None:
                clean_w_t = torch.ones(y_clean_t.size(0), dtype=torch.float32)
            else:
                _cw = np.asarray(_cw, dtype=np.float64).reshape(-1)
                if _cw.size != y_clean_t.size(0):
                    raise ValueError(
                        f"clean-set weights length {_cw.size} does not match "
                        f"clean set size {y_clean_t.size(0)}")
                clean_w_t = torch.as_tensor(_cw, dtype=torch.float32)

        # Validation set
        has_val = (X_val is not None and y_val_noisy is not None)
        if has_val:
            assert X_val is not None and y_val_noisy is not None
            z_vae_val = _encode_vae_latents(vae, X_val, device, batch_size,
                                            accel=self.accel).cpu()
            X_val_t = torch.as_tensor(
                np.asarray(X_val).reshape(len(X_val), -1), dtype=torch.float32)
            X_val_4d = None
            if is_image_data:
                xv = np.asarray(X_val)
                if xv.ndim == 4:
                    X_val_4d = torch.as_tensor(
                        xv, dtype=torch.float32).permute(0, 3, 1, 2)
            yhat_val_t = torch.as_tensor(np.asarray(y_val_noisy), dtype=torch.long)
            yval_clean_t = (
                torch.as_tensor(np.asarray(y_val_clean), dtype=torch.long)
                if y_val_clean is not None else None)
        else:
            # No validation set: no checkpoint selection happens (the clean
            # clean_set is NEVER used as a validation proxy — that was optimistic).
            log.warning(
                "  [LSNPC] no validation set supplied; checkpoint selection "
                "is DISABLED (last checkpoint will be used)")
            z_vae_val = None
            X_val_t = None
            X_val_4d = None
            yhat_val_t = None
            yval_clean_t = None

        # ── Stash per-run state for train_one_epoch / eval_and_save ──
        self._val_noisy_real = bool(val_noisy_labels_are_real)
        self._z_vae_train = z_vae_train
        self._X_train_t = X_train_t
        self._X_train_4d = X_train_4d
        self._y_train_t = y_train_t
        self._has_clean_set = has_clean_set
        self._use_semi = use_semi
        if has_clean_set:
            self._z_vae_clean = z_vae_clean
            self._X_clean_t = X_clean_t
            self._X_clean_4d = X_clean_4d
            self._y_clean_t = y_clean_t
            self._yhat_clean_t = yhat_clean_t
            self._clean_w_t = clean_w_t
        self._z_vae_val = z_vae_val
        self._X_val_4d = X_val_4d
        self._X_val_t = X_val_t
        self._yhat_val_t = yhat_val_t
        self._yval_clean_t = yval_clean_t
        self._K = K
        self._batch_size = batch_size
        self._beta = float(cfg.lsnpc_beta)          # KL coefficient of the correction objective
        _eta = cfg.lsnpc_eta
        self._eta = float(_eta) if _eta is not None else 0.1
        self._loss_type = str(cfg.lsnpc_loss)
        self._ckpt_path = ckpt_path

        # ── Optimizer ───────────────────────────────────────────────
        # eps=1e-6 is REQUIRED under bf16 autocast: 1e-8 lets bf16
        # gradient noise into the update when grads are near zero.
        # Discriminative fine-tuning: the backbone may take a smaller step
        # than the correction heads. At 1.0 the single-group path is used.
        # The full backbone at the head LR loses the pretrained representation
        # (per-dim feature std ~2e-4, class-mean separation ~1.08), which
        # collapses the downstream head to majority class.
        self._bb_lr_scale = float(getattr(cfg, "lsnpc_backbone_lr_scale", 1.0))
        self.optimizer = torch.optim.AdamW(
            self._param_groups(lr), lr=lr,
            eps=float(cfg.lsnpc_eps))

        # ── DataLoader ──────────────────────────────────────────────
        # CPU tensors → tensor_loader → accel.prepare places each batch.
        train_x = X_train_4d if X_train_4d is not None else X_train_t
        train_dl = tensor_loader(train_x, y_train_t, z_vae_train,
                                 batch_size=batch_size, shuffle=True)
        # Pin the CPU tensors so the loader's host→device copies stay
        # non-blocking. pin_memory() needs a CUDA backend, so guard it.
        on_cuda = device.startswith("cuda")

        def _pin(t):
            if t is None or not on_cuda:
                return t
            try:
                return t.pin_memory()
            except (RuntimeError, AssertionError):
                # Only a missing CUDA context is legitimate; anything else
                # (e.g. OOM) must not silently degrade every H2D transfer.
                log.warning(
                    "pin_memory() failed; continuing unpinned — H2D "
                    "transfers will be synchronous",
                    exc_info=True)
                return t

        self._z_vae_train = _pin(z_vae_train)
        self._X_train_4d = _pin(X_train_4d)
        self._X_train_t = _pin(X_train_t)
        self._y_train_t = _pin(y_train_t)
        # Clean-set and val sources: pinned for the same reason.
        if self._has_clean_set:
            self._z_vae_clean = _pin(self._z_vae_clean)
            self._X_clean_4d = _pin(self._X_clean_4d)
            self._X_clean_t = _pin(self._X_clean_t)
            self._y_clean_t = _pin(self._y_clean_t)
            self._yhat_clean_t = _pin(self._yhat_clean_t)
        if self._z_vae_val is not None:
            self._z_vae_val = _pin(self._z_vae_val)
            self._X_val_4d = _pin(self._X_val_4d)
            self._X_val_t = _pin(self._X_val_t)
            self._yhat_val_t = _pin(self._yhat_val_t)
            self._yval_clean_t = _pin(self._yval_clean_t)
        self._unwrap_lsnpc = accel.unwrap_model(self.model)

        # ── Clean-set loader ────────────────────────────────────────
        # shuffle=False keeps the IW pass order deterministic across epochs.
        self._clean_dl = None
        if self._has_clean_set:
            self._clean_dl = tensor_loader(
                self._X_clean_4d if self._X_clean_4d is not None
                else self._X_clean_t,
                self._y_clean_t,
                self._yhat_clean_t,
                self._z_vae_clean,
                self._clean_w_t,
                batch_size=batch_size, shuffle=False)

        # ── Validation loader ───────────────────────────────────────
        # Built once on CPU; accel.prepare places the batches.
        self._val_dl = None
        x_val_cpu = (
            self._X_val_4d if self._X_val_4d is not None
            else self._X_val_t
        )
        if (x_val_cpu is not None and self._z_vae_val is not None
                and self._yhat_val_t is not None):
            val_tensors: list[torch.Tensor] = [
                x_val_cpu, self._z_vae_val, self._yhat_val_t
            ]
            if self._yval_clean_t is not None:
                val_tensors.append(self._yval_clean_t)
            self._val_dl = tensor_loader(*val_tensors,
                                         batch_size=batch_size, shuffle=False)

        # ── One prepare call wraps model + optimizer + all loaders ──
        prepare_args: list = [self.model, self.optimizer, train_dl]
        if self._clean_dl is not None:
            prepare_args.append(self._clean_dl)
        if self._val_dl is not None:
            prepare_args.append(self._val_dl)
        prepared = accel.prepare(*prepare_args)
        # Re-bind: model, optimizer, train_dl are always present.
        self.model, self.optimizer, train_dl = prepared[0], prepared[1], prepared[2]
        idx = 3
        if self._clean_dl is not None:
            self._clean_dl = prepared[idx]; idx += 1
        if self._val_dl is not None:
            self._val_dl = prepared[idx]; idx += 1
        # accel.prepare may wrap the model in DDP/DP.
        self._unwrap_lsnpc = accel.unwrap_model(self.model)

        # ── Encoder liveness (image runs) ───────────────────────────
        # Cheap once-per-batch forward; see _encoder_liveness for why this
        # must abort rather than warn.
        self._encoder_spread_init = _encoder_liveness(
            self._unwrap_lsnpc, next(iter(train_dl))[0], "init")
        if self._encoder_spread_init is not None:
            log.info(f"  [LSNPC] encoder mu spread (init): "
                     f"{self._encoder_spread_init:.3e}")
        self._backbone_spread_init = _backbone_liveness(
            self._unwrap_lsnpc, next(iter(train_dl))[0], "init")
        if self._backbone_spread_init is not None:
            log.info(f"  [LSNPC] backbone feature spread (init): "
                     f"{self._backbone_spread_init:.3e}")

        # ── Scheduler (cosine, match VAE pattern) ───────────────────
        self.lr_scheduler = self._make_scheduler(
            self.optimizer,
            max(1, len(train_dl)) * epochs,
        )

        # ── Checkpoint selection state ──────────────────────────────
        self._best_score = -float("inf")
        self._best_state = None
        self._best_desc = ""
        self.history: dict[str, list] = {
            "epoch": [], "loss": [], "recon": [], "kl_zhat": [], "kl_z": [],
            "clean_set_loss": [], "val_err_h": [],
            "val_err_noisy": [], "epoch_time_s": [],
        }

        # ── Run ─────────────────────────────────────────────────────
        self.train_model(epochs, train_dl, verbose=True)

        # ── Post-training ───────────────────────────────────────────
        self.model = accel.unwrap_model(self.model)
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            log.info(f"  [LSNPC] restored best checkpoint ({self._best_desc})")

        if ckpt_path is not None:
            torch.save(self.model.state_dict(), ckpt_path)
            log.info(f"  [LSNPC] saved best checkpoint to {ckpt_path}")
        self.model.eval()
        return self

    # ── Epoch loop ──────────────────────────────────────────────────
    def train_one_epoch(self, train_loader: Any, **kwargs: Any) -> float:
        """Two-phase training: Phase A (unsupervised) + Phase B (semi).
        Follows the VAE trainer pattern: zero_grad → forward → backward →
        clip_grad → optimizer.step → scheduler.step."""
        _t0 = time.perf_counter()

        # Forward KL warm-up epoch to LSNPC model.
        if hasattr(self._unwrap_lsnpc, "set_current_epoch"):
            self._unwrap_lsnpc.set_current_epoch(self._current_epoch)

        # ── Phase A: unsupervised pass on the noisy set ──
        agg = {"loss": 0.0, "recon": 0.0, "recon_label": 0.0,
               "kl_zhat": 0.0, "kl_z": 0.0}
        n_batches = 0
        is_main = (self.accel is None) or self.accel.is_main_process
        total_ep = self._n_epochs
        ep_tag = f"{self._current_epoch}/{total_ep}"
        pbar = tqdm.tqdm(
            train_loader,
            disable=not is_main,
            desc=f"epoch {ep_tag} | loss --",
        )
        for xi, yi, zvi in pbar:
            # Batches arrive on device (DataLoaderShard from accel.prepare);
            # no index round-trip, no manual .to().
            with self.accel.autocast():
                _eta = self._eta if self._use_semi else 0.0
                losses_dict = self._unwrap_lsnpc.compute_loss(
                    xi, yi, y_clean=None, beta=self._beta, eta=_eta,
                    z_vae=zvi,
                )
            self.optimizer.zero_grad()
            self.accel.backward(losses_dict["loss"])
            self._clip_grad(self.model)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            agg["loss"] += float(losses_dict["loss"].item())
            for k in ("recon", "recon_label", "kl_zhat", "kl_z"):
                v = losses_dict.get(k)
                if v is None:
                    continue
                agg[k] += (float(v.detach().item()) if hasattr(v, "detach")
                           else float(v))
            n_batches += 1
            if is_main:
                pbar.set_description(
                    f"epoch {ep_tag} | loss "
                    f"{agg['loss'] / max(1, n_batches):.4f}")

        # ── Phase B: semi-supervised pass on the clean set ──
        # Batched so a large clean set (the full test set in data mode) fits
        # in memory: the optimizer steps per batch, the scheduler per epoch.
        clean_set_val = 0.0
        if self._has_clean_set and self._use_semi:
            clean_sum = 0.0
            n_clean_batches = 0
            clean_pbar = tqdm.tqdm(
                self._clean_dl,
                disable=not is_main,
                desc=f"epoch {ep_tag} | clean set | loss --",
                leave=False,
            )
            for xb, yb, yhb, zvb, wb in clean_pbar:
                with self.accel.autocast():
                    losses_semi = self._unwrap_lsnpc.compute_loss(
                        xb, yhb, y_clean=yb, beta=self._beta, eta=self._eta,
                        z_vae=zvb, K_clean_set=self._K,
                        clean_row_weight=wb,
                    )
                self.optimizer.zero_grad()
                self.accel.backward(losses_semi["loss"])
                self._clip_grad(self.model)
                self.optimizer.step()
                clean_sum += float(losses_semi["clean_set_loss"].item())
                n_clean_batches += 1
                if is_main:
                    clean_pbar.set_description(
                        f"epoch {ep_tag} | clean set | loss "
                        f"{clean_sum / max(1, n_clean_batches):.4f}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            clean_set_val = clean_sum / max(1, n_clean_batches)

        # Store per-epoch states used by eval_and_save logging.
        self._agg = agg
        self._n_batches = n_batches
        self._clean_set_val = clean_set_val
        self._epoch_time_s = time.perf_counter() - _t0
        return agg["loss"] / max(1, n_batches)

    # ── Validation + checkpointing ───────────────────────────────────
    def eval_and_save(self, epoch, n_epochs, val_loader=None,
                      test_loader=None, verbose=False):
        """Compute validation metrics and update best checkpoint."""
        lsnpc = self.model
        if self.accel is not None:
            lsnpc = self.accel.unwrap_model(self.model)
        if hasattr(lsnpc, "module"):
            lsnpc = lsnpc.module

        # A fine-tuned backbone collapses progressively; catch it at the first
        # epoch rather than after ten epochs of degenerate rows.
        if self.train_loader is not None:
            _batch_x = next(iter(self.train_loader))[0]
            _spread = _encoder_liveness(lsnpc, _batch_x, f"epoch {epoch + 1}")
            _init = getattr(self, "_encoder_spread_init", None)
            if (_spread is not None and _init not in (None, 0.0)
                    and _spread < _init * 1e-2):
                log.warning(
                    "  [LSNPC] encoder mu spread has fallen %.1fx since "
                    "initialisation (%.3e -> %.3e): the representation is "
                    "collapsing .",
                    _init / max(_spread, 1e-30), _init, _spread)
            _bspread = _backbone_liveness(lsnpc, _batch_x, f"epoch {epoch + 1}")
            _binit = getattr(self, "_backbone_spread_init", None)
            if (_bspread is not None and _binit not in (None, 0.0)
                    and _bspread < _binit * 1e-2):
                log.warning(
                    "  [LSNPC] backbone feature spread has fallen %.1fx since "
                    "initialisation (%.3e -> %.3e): the residual stream is "
                    "collapsing .",
                    _binit / max(_bspread, 1e-30), _binit, _bspread)

        val_metrics = {"val_err_h": float("nan"),
                       "val_err_noisy": float("nan")}
        if self._val_dl is not None:
            val_metrics = _lsnpc_val_metrics(
                lsnpc, self._val_dl, self._K, accel=self.accel,
            )
            if not getattr(self, "_val_noisy_real", True):
                # The val's noisy labels are a placeholder (encoder input only),
                # not a real annotation stream: the noisy-label metrics are
                # undefined here, never 0.
                val_metrics["val_err_noisy"] = float("nan")
                val_metrics["val_f1_noisy"] = float("nan")
            # Best-checkpoint selection is on val_err_h, for EVERY run: it is
            # the one validation metric defined whether or not the noisy labels
            # are a real annotation stream.
            val_err_h = val_metrics.get("val_err_h")
            if val_err_h is not None and not np.isnan(val_err_h):
                current_score = -val_err_h
                desc = f"val_err_h={val_err_h:.4f}"
                if current_score > self._best_score:
                    self._best_score = current_score
                    self._best_desc = desc
                    self._best_state = copy.deepcopy(lsnpc.state_dict())

        # ── Per-epoch logging ─────────────────────────────────────
        agg = self._agg
        n_b = max(1, self._n_batches)
        clean_set_val = self._clean_set_val

        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(agg["loss"] / n_b)
        self.history["recon"].append(agg["recon"] / n_b)
        self.history["kl_zhat"].append(agg["kl_zhat"] / n_b)
        self.history["kl_z"].append(agg["kl_z"] / n_b)
        self.history["clean_set_loss"].append(clean_set_val)
        self.history["val_err_h"].append(val_metrics["val_err_h"])
        self.history["val_err_noisy"].append(val_metrics["val_err_noisy"])
        # Wall-clock seconds for this epoch (Phase A + Phase B), recorded by
        # train_one_epoch.
        self.history["epoch_time_s"].append(self._epoch_time_s)

        if verbose or (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (f"  [LSNPC] epoch {epoch + 1}/{n_epochs} "
                   f"loss={agg['loss'] / n_b:.4f} "
                   f"recon_lbl={agg['recon_label'] / n_b:.4f} "
                   f"kl_zhat={agg['kl_zhat'] / n_b:.4f} "
                   f"kl_z={agg['kl_z'] / n_b:.4f}")
            et = self._epoch_time_s
            msg += f" epoch_time={et:.2f}s"
            if self._has_clean_set:
                msg += f" clean_set={clean_set_val:.4f}"
            if self._z_vae_val is not None:
                vfh = val_metrics.get("val_f1_h")
                vfn = val_metrics.get("val_f1_noisy")
                msg += (f" val_err_h={val_metrics['val_err_h']:.4f}"
                        f" val_err_noisy={val_metrics['val_err_noisy']:.4f}")
                if vfh is not None and not np.isnan(vfh):
                    msg += (f" val_f1_h={vfh:.4f}"
                            f" val_f1_noisy={vfn:.4f}")
            log.info(msg)
