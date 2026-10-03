"""Train the universal plausibility VAE.

Class-agnostic autoencoder used as the canonical plausibility scorer for all
methods. Cached on disk keyed by
`(dataset, noise_type, noise_rate, latent_dim, img_size, encoder, decoder)` —
auto-loaded if a matching checkpoint exists.

`VAETrainer` owns the model-agnostic epoch loop: it calls
`model.train_step(batch)` per batch, so each VAE keeps its own forward+loss
signature while the trainer owns the optimizer, scheduler, grad-clip and
Accelerate integration.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from data_process.image_loader import build_image_loader
from models.vae import VAE
from trainers.base import AccelLike, Trainer
from utils.batching import tensor_loader

if TYPE_CHECKING:  # avoids the import cycle; never executed at runtime
    from experiments.arguments import ExperimentConfig

log = logging.getLogger(__name__)

DEFAULT_CKPT_DIR = Path("results/ckpt")
DEFAULT_NOISE_SCALE = 0.01


# ── Model-agnostic VAE trainer ─────────────────────────────────────────


class VAETrainer(Trainer):
    """Universal VAE trainer. Works with any model exposing
    `train_step(batch) -> (loss, recon, kl)`."""

    def __init__(self, *args, early_stop_patience=20, val_split=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_patience = early_stop_patience
        self.val_split = val_split
        self.log_interval = int(getattr(self.config, "vae_log_interval", 20))

    def train(self, tensors: tuple[torch.Tensor, ...],
              n_epochs: int, batch_size: int = 256, lr: float = 1e-3,
              noise_scale: float | None = None,
              verbose: bool = False,
              use_augment: bool = False,
              crop_pad: int = 4) -> "VAETrainer":
        """Train `self.model` on the given tensor tuple.

        The model's `train_step` knows how to unpack the batch. For an
        unconditional VAE `tensors=(X, target)`; for a conditional VAE
        `tensors=(X, c_soft, target)`.

        Automatically splits off a validation set for early stopping.

        With `use_augment=True` and 4-D (NHWC) tensors the loaders come from
        ``data_process.image_loader``; otherwise from ``utils.batching``. Both
        go through the same ``accel.prepare`` below.
        """
        accel = self.accel
        n_total = len(tensors[0])
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val

        if use_augment and tensors[0].ndim == 4:
            X = tensors[0].numpy()
            noise = float(noise_scale or 0.0)
            if n_val > 0 and n_val < n_total:
                train_dl = build_image_loader(
                    X[:n_train], batch_size, shuffle=True, train=True,
                    crop_pad=crop_pad, noise_scale=noise)
                val_dl = build_image_loader(
                    X[n_train:], batch_size, shuffle=False, train=False,
                    crop_pad=0, noise_scale=noise)
            else:
                train_dl = build_image_loader(
                    X, batch_size, shuffle=True, train=True,
                    crop_pad=crop_pad, noise_scale=noise)
                val_dl = None
        else:
            if noise_scale is not None and noise_scale > 0:
                target = tensors[-1]
                tensors = (*tensors[:-1], target + noise_scale * torch.randn_like(target))

            # In-memory tensor columns: workers would only add IPC. Split for
            # early stopping.
            if n_val > 0 and n_val < n_total:
                train_dl = tensor_loader(
                    *(t[:n_train] for t in tensors),
                    batch_size=batch_size, shuffle=True)
                val_dl = tensor_loader(
                    *(t[n_train:] for t in tensors),
                    batch_size=batch_size, shuffle=False)
            else:
                train_dl = tensor_loader(*tensors, batch_size=batch_size,
                                         shuffle=True)
                val_dl = None

        self._best_val_loss = float('inf')
        self._patience_counter = 0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, eps=1e-6,
        )
        # `accel.prepare` places the loaders' batches (DataLoaderShard), so the
        # epoch loops never touch `.to(device)`.
        if val_dl is not None:
            self.model, self.optimizer, train_dl, val_dl = accel.prepare(
                self.model, self.optimizer, train_dl, val_dl)
        else:
            self.model, self.optimizer, train_dl = accel.prepare(
                self.model, self.optimizer, train_dl)
        self._val_dl = val_dl
        # Patience defaults to n_epochs/10.
        if not hasattr(self.config, "early_stop_patience") or self.config.early_stop_patience <= 0:
            self.early_stop_patience = max(1, n_epochs // 10)
        self.lr_scheduler = self._make_scheduler(
            self.optimizer, max(1, len(train_dl)) * n_epochs,
        )
        self.train_model(n_epochs, train_dl, val_loader=val_dl, verbose=verbose)
        self.model = accel.unwrap_model(self.model).to(self.device).eval()
        return self

    def train_one_epoch(self, train_loader: Any, **kwargs: Any) -> float:
        """One epoch: call `model.train_step(batch)` per batch.

        ``train_loader`` is already ``accel.prepare``-d, so its batches arrive
        on the right device; the loop does no placement of its own."""
        total_loss = 0.0
        n_batches = 0
        self.train_mode()
        # Forward KL warm-up epoch to model.
        if hasattr(self.model, "set_current_epoch"):
            self.model.set_current_epoch(self._current_epoch)
        for batch in train_loader:
            with self.accel.autocast():
                loss, _, _ = self.model.train_step(batch)
            self.optimizer.zero_grad()
            self.accel.backward(loss)
            self._clip_grad(self.model)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def eval_and_save(self, epoch, n_epochs, val_loader=None,
                      test_loader=None, verbose=False):
        """Compute validation loss and trigger early stopping."""
        if val_loader is None:
            return
        self.eval_mode()
        total_val_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0
        with torch.no_grad():
            with self.accel.autocast():
                for batch in val_loader:
                    val_loss, recon, kl = self.model.train_step(batch)
                    total_val_loss += float(val_loss.item())
                    total_recon += float(recon.item()) if isinstance(recon, torch.Tensor) else float(recon)
                    total_kl += float(kl.item()) if isinstance(kl, torch.Tensor) else float(kl)
                    n_batches += 1
        val_loss = total_val_loss / max(1, n_batches)
        # Log every `log_interval` epochs (plus first/last) to keep the
        # training log readable; early-stopping bookkeeping still runs
        # every epoch.
        log_now = (epoch + 1) % self.log_interval == 0 \
            or epoch == 0 or epoch == n_epochs - 1
        if log_now:
            if verbose:
                recon_avg = total_recon / max(1, n_batches)
                kl_avg = total_kl / max(1, n_batches)
                # Report active latent units if available.
                active_info = ""
                if hasattr(self.model, "_mu_running") and hasattr(self.model, "_count") and self.model._count > 0:
                    logvar = self.model._logvar_running
                    active = int((logvar.exp() > 0.01).sum())
                    active_info = f"  active={active}/{self.model.latent_dim}"
                log.info(f"  val_loss={val_loss:.4f}  recon={recon_avg:.4f}  kl={kl_avg:.4f}{active_info}")
            else:
                log.info(f"  val_loss={val_loss:.4f}")

        if val_loss < self._best_val_loss - 1e-6:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.early_stop_patience:
                self._early_stopped = True


# ── Universal plausibility VAE: construction + on-disk cache ───────────


def train_universal_vae(
    X_train: np.ndarray,
    config: ExperimentConfig,
    device: str,
    accel: AccelLike | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    latent_dim: int | None = None,
    hidden_dims: Sequence[int] | None = None,
    lr: float = 1e-4,
    verbose: bool = True,
    use_augment: bool = True,
) -> tuple[torch.nn.Module, dict]:
    """Train a fresh universal plausibility VAE.

    Targets are `X_train` itself (autoencoder). No class conditioning.
    Delegates the epoch loop to the model-agnostic `VAETrainer`.

    Returns:
        `(model, arch_meta)` where `model` is the trained VAE in eval
        mode and `arch_meta` is a dict describing the architecture (so
        the state_dict can be re-loaded later).
    """
    if latent_dim is None:
        latent_dim = config.latent_dim
    if hidden_dims is None:
        h = config.hidden_dim
        hidden_dims = (h, h // 2)
    batch_size = config.batch_size
    n_epochs = config.vae_epochs

    # ImageDataset stores CHW; the augment loader expects NHWC. Detect the
    # layout rather than assuming NHWC (a CHW CIFAR batch read as NHWC gives
    # H=3/W=32/C=32 and crashes the conv stack — fixed previously).
    if X_train.ndim == 4:
        if X_train.shape[1] in (1, 3) and X_train.shape[3] not in (1, 3):
            X_train = X_train.transpose(0, 2, 3, 1)  # CHW → NHWC
        H, W, C = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    elif X_train.ndim == 3:
        H, W, C = X_train.shape[1], X_train.shape[2], 1
    else:
        raise ValueError(f"Unsupported image shape: {X_train.shape}")
    log.info(f"Building Conv VAE for {C}\\u00d7{H}\\u00d7{W} images")
    vae_beta = float(getattr(config, "image_vae_beta", 1.0))
    model = VAE(
        latent_dim=latent_dim,
        beta=vae_beta,
        img_channels=C,
        img_size=max(H, W),
    )
    kind = "ConvVAE"
    _img_channels = C
    _img_size = max(H, W)

    X_t = torch.as_tensor(X_train)  # CPU; the loader places each batch
    trainer = VAETrainer(model, config=config, device=device, accel=accel)
    trainer.train(
        tensors=(X_t, X_t),
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        noise_scale=noise_scale,
        verbose=verbose,
        use_augment=use_augment,
    )

    arch_meta = {
        "kind": kind,
        "input_dim": X_train.shape[1],
        "latent_dim": latent_dim,
        "hidden_dims": list(hidden_dims),
        "noise_scale": float(noise_scale),
    }
    arch_meta["img_channels"] = _img_channels
    arch_meta["img_size"] = _img_size
    return trainer.model, arch_meta


# ── Checkpoint helpers ────────────────────────────────────────────────


def ckpt_path(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    ckpt_dir: os.PathLike = DEFAULT_CKPT_DIR,
    latent_dim: int | None = None,
    img_size: int | None = None,
) -> Path:
    """Deterministic checkpoint path for the universal plausibility VAE.

    `latent_dim` and `img_size` are part of the key so runs with a different
    latent dim or input resolution never load a mismatched model.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ld = f"_ld{latent_dim}" if latent_dim is not None else ""
    sz = f"_sz{img_size}" if img_size is not None else ""
    fname = (
        f"universal_vae_{dataset}_{noise_type}{int(noise_rate * 100)}{ld}{sz}.pt"
    )
    return ckpt_dir / fname


def _build_model(arch_meta: dict) -> torch.nn.Module:
    """Construct a fresh model from a saved `arch_meta` dict.

    Only the conv VAE is rebuildable. The tabular VAE that wrote the older
    ``kind="TabularVAE"`` bundles was removed with the tabular code path, so
    those checkpoints are stale and are rejected by the guard below.
    """
    kind = arch_meta["kind"]
    if kind == "ConvVAE":
        return VAE(
            latent_dim=arch_meta["latent_dim"],
            beta=1.0,
            img_channels=arch_meta.get("img_channels", 3),
            img_size=arch_meta.get("img_size", 32),
        )
    raise ValueError(f"Unknown VAE kind in arch_meta: {kind!r}")


def save_state(ckpt_path: Path, model: torch.nn.Module, arch_meta: dict) -> None:
    """Persist a VAE as a `state_dict` + arch metadata bundle."""
    torch.save({"state_dict": model.state_dict(), "arch": dict(arch_meta)},
               ckpt_path)


def load_state(ckpt_path: Path, device: str) -> torch.nn.Module:
    """Reconstruct a VAE from a checkpoint saved via `save_state`."""
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    if (
        not isinstance(bundle, dict)
        or "state_dict" not in bundle
        or "arch" not in bundle
    ):
        raise ValueError(
            f"{ckpt_path} is not a VAE state checkpoint "
            f"(expected {{'state_dict', 'arch'}})."
        )
    model = _build_model(bundle["arch"])
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval()


def get_or_train(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    X_train: np.ndarray,
    config: ExperimentConfig,
    device: str,
    accel: AccelLike | None = None,
    ckpt_dir: os.PathLike = DEFAULT_CKPT_DIR,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    force_retrain: bool = False,
    latent_dim: int | None = None,
    verbose: bool = True,
    use_augment: bool = True,
) -> torch.nn.Module:
    """Return the cached plausibility VAE, training one if absent.

    Lookup is keyed by `(dataset, noise_type, noise_rate, latent_dim)` so
    different noise settings and latent dims get independent surrogates.
    """
    if latent_dim is None:
        latent_dim = int(getattr(config, "latent_dim", 32))
    # Resolution is data-derived for image mode (conv VAE adapts to H,W);
    # include it in the cache key so 32px/64px runs never share a checkpoint.
    xa = np.asarray(X_train)
    img_size = None
    if xa.ndim == 4 and xa.shape[1] in (1, 3):
        img_size = int(max(xa.shape[2], xa.shape[3]))
    elif xa.ndim == 4:
        img_size = int(max(xa.shape[1], xa.shape[2]))
    # Explicit checkpoint override (--vae-ckpt): load it and skip training
    # entirely. Wins over both the per-cell cache and --force-retrain.
    vae_ckpt = getattr(config, "vae_ckpt", None)
    if vae_ckpt:
        ckpt = Path(vae_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"--vae-ckpt path does not exist: {ckpt}")
        if verbose:
            log.info(
                f"Loading universal plausibility VAE from --vae-ckpt {ckpt} "
                f"(training muted)"
            )
        return load_state(ckpt, device)
    path = ckpt_path(
        dataset,
        noise_type,
        noise_rate,
        ckpt_dir,
        latent_dim=latent_dim,
        img_size=img_size,
    )
    if (not force_retrain) and path.exists():
        if verbose:
            log.info(f"Loading cached universal plausibility VAE from {path}")
        return load_state(path, device)

    if verbose:
        log.info(
            f"Training universal plausibility VAE "
            f"(N={len(X_train)}, D={X_train.shape[1]}, noise_scale={noise_scale})…"
        )
    model, arch_meta = train_universal_vae(
        X_train=X_train,
        config=config,
        device=device,
        accel=accel,
        noise_scale=noise_scale,
        latent_dim=latent_dim,
        verbose=verbose,
        use_augment=use_augment,
    )
    save_state(path, model, arch_meta)
    if verbose:
        log.info(f"Saved universal plausibility VAE to {path}")
    return model
