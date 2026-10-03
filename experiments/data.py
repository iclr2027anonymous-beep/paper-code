"""Data pipeline helpers — device, image dims, feature projection, LSNPC batching.

These small, single-purpose functions are extracted from the main
pipeline orchestrator to keep it focused on sequencing.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
import numpy as np
import torch

from trainers.base import AccelLike
from utils.batching import place_loader_local, tensor_loader

log = logging.getLogger(__name__)


def _select_device() -> str:
    """Return ``cuda:0`` when available, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _detect_image_dims(X: np.ndarray) -> tuple[int, int, int]:
    """Infer (height, width, channels) from a batched image array.

    Input must be 4-D, either (N, C, H, W) or (N, H, W, C). A 2-D array is
    rejected rather than guessed at: recovering (H, W, C) from a flattened row
    count is ambiguous (3072 is 32x32x3, but isqrt would call it 55x55x1), so a
    wrong guess here would resurface much later as a confusing ConvClassifier
    shape error.
    """
    s = X.shape
    if len(s) != 4:
        raise ValueError(
            f"expected 4-D image data (N, C, H, W) or (N, H, W, C), "
            f"got shape {s}"
        )
    if s[3] in (1, 3):
        return int(s[1]), int(s[2]), int(s[3])  # NHWC
    return int(s[2]), int(s[3]), int(s[1])  # NCHW


def _carve_eval_split(rng, X_train, y_train_noisy, y_train_clean,
                      remaining_idx):
    """Carve a ~10% (≤500 rows) eval split for checkpoint selection.

    Splits ``remaining_idx`` (train rows NOT used as the clean set)
    into a training part and a small held-out eval part.  The eval split's
    clean labels are used for SELECTION only; the rows are excluded from
    training, so the clean-label budget stays at the clean-set rows.

    Returns ``(train_idx, val_idx, X_val, y_val_noisy_raw, y_val_clean_raw)``
    where the val elements are ``None`` when no split could be carved.
    """
    target_val = max(1, int(np.floor(0.10 * len(X_train))))
    n_val = min(500, target_val, len(remaining_idx))
    if n_val <= 0:
        return remaining_idx, None, None, None, None
    val_idx = rng.choice(remaining_idx, size=n_val, replace=False)
    train_idx = np.setdiff1d(remaining_idx, val_idx)
    # Fancy indexing already returns fresh arrays, so no defensive copy.
    return (train_idx, val_idx, X_train[val_idx],
            y_train_noisy[val_idx], y_train_clean[val_idx])


def _make_lsnpc_x_tensor(
    X: np.ndarray, is_image: bool,
) -> torch.Tensor:
    """Build an LSNPC-compatible x tensor on CPU (NHWC→NCHW for image data)."""
    a = np.asarray(X)
    if is_image and a.ndim == 4:
        return torch.as_tensor(a, dtype=torch.float32).permute(0, 3, 1, 2)
    return torch.as_tensor(a.reshape(len(a), -1), dtype=torch.float32)


def _batched_lsnpc_infer(
    lsnpc,
    X_tensor: torch.Tensor,
    z_vae: torch.Tensor,
    yhat_tensor: torch.Tensor,
    batch_size: int,
    device: str,
    M: int,
    *,
    accel: AccelLike | None = None,
    x_pool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run LSNPC inference in batches to avoid OOM.

    Batches ``(X, z_vae, yhat)`` with the shared factory (``utils.batching``)
    and ``place_loader``. Returns (z_corrected, c_corrected) on ``device``,
    one row per input row, in input order.

    The loader is placed with ``accel_like=None`` on purpose even when
    ``accel`` is given: ``accel.prepare`` shards a map-style loader across
    processes (``BatchSamplerShard``), so on a multi-process run it would
    yield each rank's SLICE of the rows. Every caller assembles the batches
    back into the caller's row order (``torch.cat``, and ``x_pool`` sliced by
    running offset), which a shard would silently corrupt: the shard is a
    valid tensor of the wrong rows. ``accel`` is used for autocast only.

    ``x_pool``, when given, is the (n, feat) pooled image feature for the
    whole input, sliced per batch and handed to the model so the frozen
    trunk is not re-run; it must have been produced under the same precision
    context the model would have run in (these scorers do not autocast).
    """
    idx_dl = place_loader_local(
        tensor_loader(X_tensor, z_vae, yhat_tensor, batch_size=batch_size),
        device)
    z_parts: list[torch.Tensor] = []
    c_parts: list[torch.Tensor] = []
    amp = (accel.autocast() if accel is not None else nullcontext())
    i0 = 0
    with torch.no_grad():
        with amp:
            for xb, zv, yh in idx_dl:
                pool = None if x_pool is None else x_pool[i0:i0 + xb.shape[0]]
                i0 += xb.shape[0]
                zb = lsnpc.sample_corrected_latent(xb, zv, yh, M=M, x_pool=pool)
                z_parts.append(zb.detach().float())
                cb = lsnpc.corrected_conditioning(xb, zb, x_pool=pool)
                c_parts.append(cb.detach().float())
    return torch.cat(z_parts, dim=0), torch.cat(c_parts, dim=0)
