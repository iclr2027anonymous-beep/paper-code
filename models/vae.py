"""Gaussian VAE over image data.

The components themselves (ConvEncoder, ShuffleResDecoder) live in
``models/archs.py``; this module composes them and owns the loss.
bf16 + HuggingFace compatible.
"""

import torch
import torch.nn as nn

from .archs import ConvEncoder, ShuffleResDecoder


def class_to_onehot(
    y: torch.Tensor,
    n_classes: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Convert a class-index tensor to a one-hot (or one-column-1) tensor.

    Equivalent to:
        c = torch.zeros(y.size(0), n_classes, device=device)
        c.scatter_(1, y.unsqueeze(1), 1.0)

    Args:
        y: Long tensor of class indices, shape (N,). Values must be in
            [0, n_classes).
        n_classes: number of classes (C).
        device: optional device for the output tensor. If `None`, defaults
            to the device of `y`.

    Returns:
        Float tensor of shape (N, n_classes) with a 1.0 in the column
        corresponding to each class index.
    """
    if device is None:
        device = y.device
    y = y.to(device=device, dtype=torch.long).view(-1)
    c = torch.zeros(y.size(0), n_classes, device=device)
    c.scatter_(1, y.unsqueeze(1), 1.0)
    return c


def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        beta=1.0,
        img_channels=3,
        img_size=32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Runtime-selected families share no base: pin the attribute type.
        self.encoder: nn.Module
        self.decoder: nn.Module

        self.encoder = ConvEncoder(latent_dim, img_channels, img_size)
        self.decoder = ShuffleResDecoder(latent_dim, img_channels, img_size)

        # Track posterior statistics for diagnostics.
        self.register_buffer("_mu_running", torch.zeros(latent_dim), persistent=False)
        self.register_buffer(
            "_logvar_running", torch.zeros(latent_dim), persistent=False
        )
        self.register_buffer("_count", torch.zeros(1), persistent=False)

    def encode(self, x):
        x = _to_nchw_if_nhwc(x)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).permute(0, 2, 3, 1)  # NCHW → NHWC

    def reparameterize(self, mu, logvar):
        return reparameterize(mu, logvar)

    def forward(self, x):
        x = _to_nchw_if_nhwc(x)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon.permute(0, 2, 3, 1), mu, logvar, z  # NCHW → NHWC

    def compute_loss(self, x, x_recon, mu, logvar):
        # Per-sample pixel MSE (sum over pixels, mean over batch)
        per_sample_recon = torch.pow(x_recon - x, 2).reshape(x.size(0), -1).sum(dim=-1)
        recon_loss = per_sample_recon.mean()

        # Per-latent-dim KL
        raw_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
        per_dim_kl = raw_kl.sum(dim=-1)  # (B,) sum over latent dims

        kl_loss = per_dim_kl.mean()

        # Update running posterior statistics.
        with torch.no_grad():
            self._mu_running.mul_(0.99).add_(mu.mean(dim=0), alpha=0.01)
            self._logvar_running.mul_(0.99).add_(logvar.mean(dim=0), alpha=0.01)
            self._count += 1

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def train_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward + loss for one batch. `batch = (x, target)`.

        Returns `(loss, recon_loss, kl_loss)`. The trainer calls this
        so it doesn't need to know the model's forward signature.
        """
        x, target = batch
        x_recon, mu, logvar, z = self.forward(x)
        return self.compute_loss(target, x_recon, mu, logvar)

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Per-dim log p(x) under this VAE's decoder.

        Decodes the posterior mean and returns the per-dim Gaussian
        log-likelihood (unit observation noise) with the channel axis
        averaged out: ``-0.5 * (x - x_recon)^2``. Shape ``(B, H, W)`` for
        images, ``(B, D)`` for features.
        """
        mu_z, _ = self.encode(x)
        x_recon = self.decode(mu_z)
        per_dim = -0.5 * torch.pow(x - x_recon, 2).mean(dim=-1)
        return per_dim

    # Alias so `Trainer.predict()` can delegate uniformly across models.
    predict = log_prob


def _to_nchw_if_nhwc(x):
    """Return ``x`` as NCHW when it is an NHWC image batch; otherwise unchanged."""
    if x.ndim == 4 and x.shape[-1] in (1, 3):
        return x.permute(0, 3, 1, 2)  # NHWC → NCHW
    return x
