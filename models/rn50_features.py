"""Frozen RN50 feature encoder for LSNPC image correction (data-level).

Mirrors ``TextEncoder``'s contract: ``encode(x)`` returns the FROZEN
ImageNet feature embedding (2048-d) as ``(mu, None)`` — no random mu/logvar
projection heads. The frozen features play the same role as the mpnet
embeddings in the text pipeline: the embedding space IS the LSNPC latent
``z``; LSNPC recenters it toward the clean class; the MLP data-decoder
reconstructs the embedding.

Input: CIFAR-normalized (N, C, H, W) float32 images; preprocessing to
ImageNet normalization + 224x224 resize happens inside the backbone path
(the same ``_pretrained_preprocess`` the ResNet50Encoder uses).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

from .archs import _pretrained_preprocess  # shared CIFAR->ImageNet conversion

_LABEL_LOGVAR_CLAMP = 10.0


class IdentityFeatureEncoder(nn.Module):
    """Identity over already-computed frozen features (the VAE-latent role).

    After the pixel encoder has produced the 2048-d features once, the
    trainer re-encodes each batch — but the features ARE the latent, so
    ``encode(x) = x`` exactly like ``TextEncoder.encode``.
    """

    def __init__(self, latent_dim: int | None = None) -> None:
        super().__init__()
        self.image_data = False
        self.latent_dim = int(latent_dim) if latent_dim is not None else 2048

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


class RN50FeatureEncoder(nn.Module):
    """Frozen ResNet-50 (ImageNet) feature extractor with the VAE-latent role.

    ``encode`` / ``forward`` return ``(feat, None)`` so the object drops into
    the trainer's ``_encode_vae_latents`` contract (unconditional VAE
    signature).  All backbone parameters are frozen; the 2048-d avg-pool
    features are the latent.
    """

    def __init__(self, latent_dim: int | None = None) -> None:
        super().__init__()
        self.pretrained = True
        self.freeze_backbone = True
        weights = tv_models.ResNet50_Weights.DEFAULT
        backbone = tv_models.resnet50(weights=weights)
        self.net = nn.Sequential(*list(backbone.children())[:-1])  # -> avgpool feat
        for p in self.net.parameters():
            p.requires_grad = False
        self.embed_dim = 2048
        self.latent_dim = int(latent_dim) if latent_dim is not None else self.embed_dim
        # No projection: the frozen features ARE the latent. Keep the
        # attribute so callers can read the latent dim.
        self.proj = nn.Identity()
        self.logvar = nn.Identity()
        self.logvar_clamp = _LABEL_LOGVAR_CLAMP
        self.image_data = False  # embedding-space: the trainer treats x as flat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        x = _pretrained_preprocess(x)
        h = self.net(x)                      # (B, 2048, 1, 1)
        return h.flatten(start_dim=1), None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.forward(x)
