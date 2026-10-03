"""Shared architectures — single source of truth.

Models import their structure from here instead of re-defining their own
stack. Membership is by *reuse*, not by modality: a component lives here as
soon as more than one model needs it.

Encoders
- ``ConvStack``   — the 3× stride-2 conv feature extractor (C→32→64→128,
                    96→12 spatial). Used by the VAE encoder and anything
                    else that needs the shared conv backbone.
- ``ConvEncoder`` — the VAE's encoder: ``ConvStack`` + mu/logvar heads.
- ``ResNet50Encoder`` / ``SwinEncoder`` / ``ViTEncoder`` — pretrained
  backbones, selected by ``build_image_encoder``.

Decoders
- ``ShuffleResDecoder`` — PixelShuffle + residual upsampling; the VAE's
  decoder.
- ``ResBlock``    — the residual block ``ShuffleResDecoder`` is built from.

Checkpoint-compatibility contract: ``ConvEncoder`` keeps its parameter
registered under the ``net`` attribute (NOT ``stack``), so VAE checkpoints
saved with the historical key layout (``encoder.net.*``) load unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from utils.paths import project_path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import SwinModel, ViTModel

from utils.pixel_transform import (
    PRETRAINED_INPUT_SIZE,
    pretrained_pixel_transform,
)

_LOGVAR_MAX = 5.0  # clamp on predicted log-variance (matches models/lsnpc)


def _pretrained_preprocess(x: torch.Tensor,
                           size: int = PRETRAINED_INPUT_SIZE) -> torch.Tensor:
    """Convert CIFAR-normalized input to ImageNet-normalized, resized to 224.

    Thin alias for ``utils.pixel_transform.pretrained_pixel_transform``.
    """
    return pretrained_pixel_transform(x, size)



class ConvStack(nn.Module):
    """Shared convolutional feature extractor for 96×96 RGB images.

    Progressive channel expansion: C → 32 → 64 → 128.
    3× stride-2 convolutions (kernel 4, pad 1) = 8× spatial reduction
    (96→12 for img_size=96). BatchNorm2d + LeakyReLU after each conv
    except the first.

    The output feature map is (B, 128, img_size//8, img_size//8); every
    consumer attaches its own head on top.
    """

    def __init__(self, img_channels: int = 3, img_size: int = 96):
        super().__init__()
        channels = [img_channels, 32, 64, 128]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            cin, cout = channels[i], channels[i + 1]
            block: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1)]
            if i > 0:
                block.append(nn.BatchNorm2d(cout))
                block.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*block))
        self.net = nn.Sequential(*blocks)
        self.feat_size = img_size // 8  # three stride-2 → /8
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEncoder(nn.Module):
    """VAE convolutional encoder (image data).

    Shared ``ConvStack`` feature extractor + linear mu/logvar heads.
    ``net`` is deliberately the attribute name (not ``stack``) so the
    state-dict keys match historical VAE checkpoints (``encoder.net.*``).
    """

    def __init__(self, latent_dim=128, img_channels=3, img_size=96):
        super().__init__()
        stack = ConvStack(img_channels, img_size)
        self.net = stack.net  # same Sequential; keeps ckpt key layout
        self.feat_size = stack.feat_size
        self.out_channels = stack.out_channels
        flat_dim = self.out_channels * self.feat_size * self.feat_size
        self.mu = nn.Linear(flat_dim, latent_dim)
        self.logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.backbone_feature(x)
        return self.mu(h), self.logvar(h)

    def backbone_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened conv-stack feature, *before* the mu/logvar heads.

        The same decomposition every other encoder exposes (see
        ``ResNet50Encoder.backbone_feature``): callers that read one image
        through several modules pool it once instead of once per module.
        """
        return self.net(x).flatten(start_dim=1)


class _PretrainedEncoder(nn.Module):
    """Shared ``forward`` of the pretrained-backbone encoders.

    A subclass owns ``backbone_feature`` (the pooled feature, before the heads)
    and the ``mu``/``logvar`` head pair; applying the heads and bounding the
    log-variance is the same everywhere, so it lives here rather than in three
    identical copies.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone_feature(x)
        return self.mu(h), self.logvar(h).clamp(max=_LOGVAR_MAX)


class ResNet50Encoder(_PretrainedEncoder):
    """VAE-style image encoder built on a torchvision ResNet50 backbone.

    Same interface as ``ConvEncoder`` (returns ``(mu, logvar)`` with
    ``latent_dim`` outputs), so any ``build_image_encoder`` consumer can swap
    backbones. ResNet50's final avg-pool features (2048-d) are projected to
    ``latent_dim`` by linear mu/logvar heads.

    ``pretrained=True`` loads ImageNet weights via ``torchvision``.

    The pipeline feeds CIFAR-10-normalized 32x32 input (the shared VAE/conv
    format). Pretrained ImageNet backbones need ImageNet-normalized 224x224
    input, so ``forward`` first converts CIFAR-normalized -> raw [0,1] ->
    resized 224x224 -> ImageNet-normalized (same canonical preprocessing as
    ``SwinEncoder`` / ``ViTEncoder``).
    """

    def __init__(self, latent_dim: int = 128, img_channels: int = 3,
                 img_size: int = 32, pretrained: bool = True,
                 freeze_backbone: bool = True, depth: int = 50):
        super().__init__()
        if img_channels != 3:
            raise ValueError(
                f"ResNet50Encoder requires 3 input channels, got {img_channels}")
        if depth not in (50, 101):
            raise ValueError(f"ResNet depth must be 50 or 101, got {depth}")
        weights = (tv_models.ResNet101_Weights.DEFAULT if depth == 101
                   else tv_models.ResNet50_Weights.DEFAULT if pretrained
                   else None)
        backbone = (tv_models.resnet101(weights=weights) if depth == 101
                    else tv_models.resnet50(weights=weights))
        # Keep conv stem -> layer4 -> avgpool (drop the classification fc).
        self.net = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
        # The norm layers follow the backbone's frozen flag. A frozen backbone
        # keeps the pretrained BatchNorm running statistics (its BN stays in
        # eval through any ``train()``). A trainable backbone fine-tunes them
        # with the rest of its weights -- the norm layers are part of the
        # backbone, so there is no separate switch for them.
        self.freeze_bn = bool(freeze_backbone)
        if self.freeze_bn:
            self._freeze_bn_stats()
        self.mu = nn.Linear(2048, latent_dim)
        self.logvar = nn.Linear(2048, latent_dim)

    def _freeze_bn_stats(self) -> None:
        """Put every BatchNorm of a frozen backbone into eval.

        Frozen means the pretrained running statistics are the reference and
        must not be overwritten by the correction distribution. It is applied
        exactly when the backbone is frozen: ``--trainable`` fine-tunes the
        norm layers too.
        """
        for m in self.net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True):
        """Re-freeze BN of a frozen backbone after any ``.train()``.

        ``lsnpc.train()`` (trainers/lsnpc.py:152) puts the whole model into
        train mode before every epoch, which would otherwise flip the
        backbone's BN back to updating. Overriding here keeps the norm mode
        tied to the backbone's frozen flag at the module that owns it, rather
        than patching each trainer call site.
        """
        super().train(mode)
        if getattr(self, "freeze_bn", False):
            self._freeze_bn_stats()
        return self

    def backbone_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled 2048-d backbone feature, *before* the mu/logvar heads.

        Exposed separately because the liveness guard has to inspect the
        residual stream itself rather than ``mu``, which the heads can
        rescale .
        """
        x = _pretrained_preprocess(x)
        h = self.net(x)  # (B, 2048, 1, 1) at 224x224 input
        return h.flatten(start_dim=1)


# Pre-downloaded image backbones, with an optional explicit location.
def _model_dir(name: str) -> str:
    """Resolve a local model without falling back to another project."""
    base = Path(os.environ.get("IMAGE_MODEL_DIR", project_path("local_models")))
    return str(base / name)


# The encoders read ``feat_dim`` from the loaded config, so Swin-Tiny's 768-d
# hidden size needs no special casing: only the path and the name differ.
_DEFAULT_SWIN_PATH = _model_dir("swin-base-patch4-window7-224")
_DEFAULT_SWIN_TINY_PATH = _model_dir("swin-tiny-patch4-window7-224")
_DEFAULT_VIT_PATH = _model_dir("vit-base-patch16-224")


class SwinEncoder(_PretrainedEncoder):
    """VAE-style image encoder built on a huggingface Swin backbone.

    Same interface as ``ConvEncoder`` (returns ``(mu, logvar)`` with
    ``latent_dim`` outputs). Loads a pre-downloaded Swin-Base or Swin-Tiny
    (``local_files_only=True``), resizes the input to the Swin-native 224x224,
    applies ImageNet normalization, and projects the pooled features
    (``hidden_size``: 1024 for Base, 768 for Tiny) to ``latent_dim`` via linear
    mu/logvar heads.
    """

    def __init__(self, latent_dim: int = 128, img_channels: int = 3,
                 img_size: int = 32, model_path: str | None = None,
                 freeze_backbone: bool = True):
        super().__init__()
        if img_channels != 3:
            raise ValueError(
                f"SwinEncoder requires 3 input channels, got {img_channels}")
        path = model_path or _DEFAULT_SWIN_PATH
        self.swin = SwinModel.from_pretrained(path, local_files_only=True)
        if freeze_backbone:
            for p in self.swin.parameters():
                p.requires_grad = False
        else:
            # A trainable backbone re-encodes the batch ~11x per step (1
            # posterior + K in iw_log_weights + K in the clean-set loop) and
            # those caches OOM a 24GB GPU at batch 32; recompute on backward
            # instead. No-op while frozen.
            self.swin.gradient_checkpointing_enable()
        self.feat_dim = int(self.swin.config.hidden_size)  # 1024 Base, 768 Tiny
        self.mu = nn.Linear(self.feat_dim, latent_dim)
        self.logvar = nn.Linear(self.feat_dim, latent_dim)

    def backbone_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled Swin feature, *before* the ``mu``/``logvar`` heads.

        Exposed for the liveness guard, which must inspect the backbone itself:
        see ``ResNet50Encoder.backbone_feature``.
        """
        x = _pretrained_preprocess(x)
        out = self.swin(x)
        pooler = getattr(out, "pooler_output", None)
        return pooler if pooler is not None else out.last_hidden_state.mean(dim=1)


class ViTEncoder(_PretrainedEncoder):
    """VAE-style image encoder built on the pre-downloaded ViT-Base/16.

    Same interface as ``ConvEncoder`` / ``SwinEncoder`` (returns
    ``(mu, logvar)`` with ``latent_dim`` outputs). Loads the reference
    project's ``vit-base-patch16-224`` (``local_files_only=True``),
    resizes the input to 224x224, applies ImageNet normalization, and
    projects pooled features to ``latent_dim`` via linear mu/logvar heads.

    Pooling is the CLS token (``last_hidden_state[:, 0]``): the checkpoint
    carries no trained pooler head (HF initializes ``pooler.dense`` fresh),
    so ``pooler_output`` must not be used. ~1.9x faster than Swin-Base per
    forward pass under bf16 (a measured reference configuration).
    """

    def __init__(self, latent_dim: int = 128, img_channels: int = 3,
                 img_size: int = 32, model_path: str | None = None,
                 freeze_backbone: bool = True):
        super().__init__()
        if img_channels != 3:
            raise ValueError(
                f"ViTEncoder requires 3 input channels, got {img_channels}")
        path = model_path or _DEFAULT_VIT_PATH
        self.vit = ViTModel.from_pretrained(path, local_files_only=True)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.feat_dim = int(self.vit.config.hidden_size)  # 768 for ViT-Base
        self.mu = nn.Linear(self.feat_dim, latent_dim)
        self.logvar = nn.Linear(self.feat_dim, latent_dim)

    def backbone_feature(self, x: torch.Tensor) -> torch.Tensor:
        """CLS feature, *before* the ``mu``/``logvar`` heads (liveness guard)."""
        x = _pretrained_preprocess(x)
        return self.vit(x).last_hidden_state[:, 0]  # CLS token (no pooler)


#: Backbones with a pretrained trunk that ``freeze_backbone`` actually
#: freezes, and that therefore build bit-identical trunks for the same name.
#: ``conv`` is absent: it has no pretrained weights and ignores the flag.
FROZEN_TRUNK_BACKBONES = frozenset(
    {"resnet50", "resnet101", "swin", "swin_tiny", "vit"})


def build_image_encoder(backbone: str, latent_dim: int,
                        img_channels: int = 3, img_size: int = 32,
                        freeze_backbone: bool = True) -> nn.Module:
    """Return the LSNPC image encoder for the named backbone.

    ``conv`` (paper ConvEncoder), ``resnet50``/``resnet101`` (torchvision,
    unmodified architecture), ``swin`` (pre-downloaded Swin-Base),
    ``swin_tiny`` (the vendored Swin-Tiny), or ``vit`` (pre-downloaded
    ViT-Base/16). All share the ``(mu, logvar)`` interface.
    ``freeze_backbone`` freezes the pretrained backbone params and with them
    its normalisation layers (heads always train); a trainable backbone
    fine-tunes its norms as well.
    """
    if backbone == "resnet50":
        return ResNet50Encoder(latent_dim, img_channels, img_size,
                               freeze_backbone=freeze_backbone)
    if backbone == "resnet101":
        return ResNet50Encoder(latent_dim, img_channels, img_size,
                               freeze_backbone=freeze_backbone,
                               depth=101)
    if backbone == "swin":
        return SwinEncoder(latent_dim, img_channels, img_size,
                           freeze_backbone=freeze_backbone)
    if backbone == "swin_tiny":
        return SwinEncoder(latent_dim, img_channels, img_size,
                           model_path=_DEFAULT_SWIN_TINY_PATH,
                           freeze_backbone=freeze_backbone)
    if backbone == "vit":
        return ViTEncoder(latent_dim, img_channels, img_size,
                          freeze_backbone=freeze_backbone)
    return ConvEncoder(latent_dim, img_channels, img_size)


# ── VAE decoder components ──────────────────────────────────────────
#
# The VAE composes these (models/vae.py); the structure lives here.


class ResBlock(nn.Module):
    """2-layer residual block: Conv-BN-ReLU-Conv-BN + shortcut, ReLU out."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.BatchNorm2d(cout),
        )
        self.shortcut = (
            nn.Conv2d(cin, cout, 1, stride)
            if cin != cout or stride != 1
            else nn.Identity()
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.net(x) + self.shortcut(x))


class ShuffleResDecoder(nn.Module):
    """PixelShuffle (sub-pixel conv) + residual upsampling decoder.

    Standalone ``decode(z)`` (no encoder skips needed): the latent is
    projected to ``start_ch`` at ``img_size/16`` and upsampled 4× with
    ``PixelShuffle(2)`` + ``ResBlock``, ending with ``Conv2d -> sigmoid``
    (NCHW ``(B, C, H, W)`` in [0, 1]). It is the VAE decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        img_channels: int = 3,
        img_size: int = 96,
        start_ch: int = 512,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.start_size = max(1, int(img_size) // 16)
        self.start_ch = int(start_ch)
        self.proj = nn.Linear(
            self.latent_dim, self.start_ch * self.start_size * self.start_size
        )
        # PixelShuffle(2) divides channels by 4: 512->128, then 128->32 ...
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.PixelShuffle(2), ResBlock(start_ch // 4, start_ch // 4)
                ),  # ->12
                nn.Sequential(
                    nn.PixelShuffle(2), ResBlock(start_ch // 16, start_ch // 4)
                ),  # ->24
                nn.Sequential(
                    nn.PixelShuffle(2), ResBlock(start_ch // 16, start_ch // 4)
                ),  # ->48
                nn.Sequential(
                    nn.PixelShuffle(2), ResBlock(start_ch // 16, start_ch // 8)
                ),  # ->96
            ]
        )
        self.out = nn.Conv2d(start_ch // 8, img_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(-1, self.start_ch, self.start_size, self.start_size)
        for up in self.ups:
            h = up(h)
        return torch.sigmoid(self.out(h))
