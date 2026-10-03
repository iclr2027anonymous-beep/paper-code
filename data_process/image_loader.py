"""Augmenting image datasets for NHWC batches.

Feeds the augmented-training path of ``trainers/plausibility_vae.py``:
random crop + horizontal flip + normalize (normalize only for validation),
with optional Gaussian target noise. Augmentation runs in ``__getitem__``, so
this is the one path where ``num_workers > 0`` pays off.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data_process.base import SizedDataset
from utils.batching import make_loader

# CIFAR-10/100 standard: input in [0, 1] after /255, then (x - 0.5)/0.5 → [-1, 1].
_IMG_MEAN = 0.5
_IMG_STD = 0.5


class _NumpyImageAugDataset(Dataset):
    """Augmenting NHWC uint8 numpy arrays on the fly.

    X is (N, H, W, C) uint8 or float32 in [0, 1]. ``train`` adds a reflect-pad
    random crop and a horizontal flip; ``crop_pad`` is the pad width. ``X`` is
    the per-sample source only; the target comes from ``_NoiseDataset``.
    """

    def __init__(self, X: np.ndarray, train: bool, crop_pad: int = 4,
                 noise_scale: float = 0.0) -> None:
        if X.dtype != np.uint8:  # one code path for both input conventions
            X = np.clip(X * 255.0, 0, 255).astype(np.uint8) if X.max() <= 1.0 \
                else np.clip(X, 0, 255).astype(np.uint8)
        self.X = X
        self.train = train
        self.crop_pad = int(crop_pad)
        self.noise_scale = float(noise_scale)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.X[idx]                                  # (H, W, C) uint8
        if self.train and self.crop_pad > 0:
            # reflect-pad, then random-crop back to (H, W)
            padded = np.pad(
                img,
                ((self.crop_pad, self.crop_pad),
                 (self.crop_pad, self.crop_pad), (0, 0)),
                mode="reflect",
            )
            H, W = img.shape[:2]
            top = np.random.randint(0, 2 * self.crop_pad + 1)
            left = np.random.randint(0, 2 * self.crop_pad + 1)
            img = padded[top:top + H, left:left + W]
            if np.random.rand() < 0.5:                       # horizontal flip
                img = np.ascontiguousarray(img[:, ::-1, :])
        # uint8 [0, 255] → float32 [-1, 1] NCHW
        out = torch.from_numpy(img).float().div_(255.0)
        out = (out - _IMG_MEAN) / _IMG_STD
        out = out.permute(2, 0, 1).contiguous()
        return out


class _NoiseDataset(Dataset):
    """Pairs a dataset with its target, adding σ N(0, 1) noise per sample."""

    def __init__(self, base: SizedDataset, target: np.ndarray | torch.Tensor,
                 noise_scale: float) -> None:
        self.base = base
        self.target = (torch.as_tensor(target)
                       if not torch.is_tensor(target) else target)
        self.noise_scale = float(noise_scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        out = self.base[idx]
        if self.noise_scale > 0:
            tgt = self.target[idx]
            if not torch.is_tensor(tgt):
                tgt = torch.as_tensor(tgt)
            tgt = tgt + self.noise_scale * torch.randn_like(tgt)
            return out, tgt
        return out, self.target[idx]


def build_image_loader(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool,
    train: bool,
    crop_pad: int = 4,
    noise_scale: float = 0.0,
    num_workers: int = 2,
) -> "torch.utils.data.DataLoader":
    """NHWC images -> (image, target) batches, built by ``make_loader``.

    ``train`` augments (crop, flip, normalize) and drops the trailing partial
    batch; otherwise the images are only normalised. ``noise_scale`` is σ on
    the target, not the image. Batches leave on CPU; ``accel.prepare`` places
    them.
    """
    img_ds = _NumpyImageAugDataset(
        X, train=train, crop_pad=crop_pad, noise_scale=noise_scale)
    # train_step unpacks `x, target = batch`; the target is the un-augmented
    # image (the VAE is an autoencoder).
    target_ds = _NoiseDataset(img_ds, torch.as_tensor(X), noise_scale)
    return make_loader(
        target_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
    )
