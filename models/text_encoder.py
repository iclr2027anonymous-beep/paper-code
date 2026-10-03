"""Text encoder for the LSNPC embedding-space correction path.

For text there is no image VAE: the frozen sentence-embedding encoder (an
mpnet-family transformer) is the encoder, and its mean-pooled sentence embedding
plays the role of the LSNPC latent ``z``. The corrected latent therefore lives in
the same ``embed_dim`` embedding space as the input.

Built on ``transformers.AutoModel`` + mean pooling over the attention mask
(all-mpnet-base-v2 style) rather than ``sentence_transformers``, which pulls in
HuggingFace ``datasets`` and collides with this repo's local ``datasets`` package.

Roles (same object, three interfaces):
  * ``encode_text(texts) -> np.ndarray``  — data-prep: embed raw sentences.
  * ``encode(x) -> (mu, logvar)``         — LSNPC "VAE" latent role: for the
    embedding-space path the frozen embedding IS the latent, so ``mu = x`` and
    ``logvar = None``. Satisfies the trainer's ``_encode_vae_latents(vae, X)``
    contract.
  * ``forward(x) -> (mu, logvar)``        — LSNPC in-model encoder role: project
    ``embed_dim -> latent_dim`` (identity when equal) with a learnable ``logvar``
    head, matching the image encoders' ``(mu, logvar)`` interface.
"""
from __future__ import annotations

from utils.paths import project_path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

_LOGVAR_MAX = 2.0

# Default local path to the downloaded sentence-embedding model.
DEFAULT_TEXT_ENCODER = (project_path('models/all-mpnet-base-v2'))


def build_text_encoder(model_path: str = DEFAULT_TEXT_ENCODER,
                       latent_dim: int | None = None,
                       freeze_backbone: bool = True) -> "TextEncoder":
    """Return a frozen ``TextEncoder`` for the named sentence-embedding model."""
    return TextEncoder(model_path, latent_dim=latent_dim,
                       freeze_backbone=freeze_backbone)


class TextEncoder(nn.Module):
    """Frozen sentence-embedding encoder for LSNPC text correction."""

    def __init__(self, model_path: str, latent_dim: int | None = None,
                 freeze_backbone: bool = True, max_length: int = 128):
        super().__init__()
        self.model_path = model_path
        self.max_length = max_length
        self.embedder = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if freeze_backbone:
            for p in self.embedder.parameters():
                p.requires_grad = False
        self.freeze_backbone = bool(freeze_backbone)
        self.embed_dim = int(self.embedder.config.hidden_size)
        self.latent_dim = int(latent_dim) if latent_dim is not None else self.embed_dim 
        self.proj = nn.Linear(self.embed_dim, self.latent_dim)
        self.logvar = nn.Linear(self.embed_dim, self.latent_dim)

    # ── data-prep interface ──────────────────────────────────────────────
    @torch.no_grad()
    def _embed_tensor(self, texts: list[str], batch_size: int = 512) -> torch.Tensor:
        """Tokenize + forward + mean-pool (mask-aware) -> (n, embed_dim)."""
        self.embedder.eval()
        if next(self.embedder.parameters()).is_cuda or not torch.cuda.is_available():
            dev = next(self.embedder.parameters()).device
        else:
            dev = "cuda"
            self.embedder = self.embedder.to(dev)
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = self.embedder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(pooled.detach().float().cpu())
        return torch.cat(outs, dim=0)

    def encode_text(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        """Embed a list of sentences -> (n, embed_dim) float32 numpy array."""
        return self._embed_tensor(texts, batch_size).numpy().astype(np.float32)

    # ── LSNPC "VAE" latent role ──────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the latent for the trainer's ``_encode_vae_latents`` contract.

        ``x`` is already the frozen sentence embedding and that embedding IS the
        LSNPC latent, so ``mu = x`` and there is no posterior variance
        (``logvar = None``).
        """
        return x, None

    # ── LSNPC in-model encoder role ──────────────────────────────────────
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the sentence embedding to ``latent_dim`` (identity if equal)."""
        h = self.proj(x)
        return h, self.logvar(x).clamp(max=_LOGVAR_MAX)
