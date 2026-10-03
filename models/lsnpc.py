"""LSNPC (Latent-Space Noise-Prediction Correction) modules (v5B).

Implements the training-time label-correction path of paper §4
(``sections_v5B/04_method.tex``).

Generative model (paper Eqs. 1–5):
    z            ~ N(0, I)                                    (latent prior) 
    ẑ | z        ~ t_ν(μ_ψ(z), diag(σ²_ψ(z)))                 (Student-t shift, generative)
    y  | z, x    ~ Cat(softmax(g_φ(x, z)))                    (clean-label predictor)
    ŷ  | ẑ, x    ~ Cat(softmax(g_φ(x, ẑ)))                    (noisy-label predictor)

Variational posterior (paper Eqs. C.2–C.4):
    q(ẑ | x, ŷ)  = t_ν(μ_θ(x, ŷ), diag(σ²_θ(x, ŷ)))          (NoisyEncoder)
    q(z | ẑ)     = N(μ_κ(ẑ),  diag(σ²_κ(ẑ)))                  (CorrectionEncoder)

The correction path ẑ → z recovers a clean latent from the noisy one.
At inference z_0 IS the LSNPC-corrected latent z, and the
conditioning signal is the corrected predictor h̃(x, ŷ) = g_φ(x, z).

**Training (v2, this file):** rather than maximising the Monte-Carlo IW-ELBO
(numerically fragile for the heavy-tailed Student-t posterior), we minimise the
closed-form CorrectionLoss surrogate of the previous implementation
(``nlc_vae.py``):

    L = recon + β · (KL_surrogate(ẑ) + KL(z))

where KL_surrogate is the exponential-family divergence
    KL(q‖p) ≈ -E[exp(logp - logq) - 1 - (logp - logq)]
computed with reparameterised Student-t samples (``D.StudentT.rsample``).
``recon`` is the label cross-entropy ``CE(g_φ(x, ẑ), ŷ)``; there is no data
reconstruction term (see ``compute_loss``).

All modules use the project's ``build_mlp`` convention (FC → RMSNorm → GELU →
Dropout), matching ``models/mlp.py``.
"""
from __future__ import annotations

import logging
import math
import random

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


from .archs import FROZEN_TRUNK_BACKBONES, build_image_encoder
from .mlp import build_mlp
from .vae import class_to_onehot

log = logging.getLogger(__name__)

# Numerical clamps.
_LOGVAR_MAX = 5.0          # clamp predicted log-variance (reference nlc_vae line 67)
_STD_MIN = 1e-4            # floor on the standard-deviation scale
_NU_MIN = 2.0              # floor on the learned df: Student-t variance is finite
_NU_MAX = 30.0             # ceiling on the learned df: beyond it the density is
                           # Gaussian to fp32 and `lgamma(nu/2)` cancellation
                           # starts to dominate the log-normaliser
_NU_INIT = 5.0             # df the learned head starts at (the former fixed df)


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                        gamma: float = 2.0,
                        alpha: float = 0.25,
                        reduction: str = "mean") -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017), one-vs-all form.

    Runs ``torchvision.ops.sigmoid_focal_loss`` over a one-hot target: every
    class contributes an independent binary focal term,
    ``alpha_t * (1 - p_t)^gamma * -log p_t``, and the terms are summed. The
    caller therefore keeps passing a class *index* and gets one loss per row,
    the shape the importance-weighted clean-set term multiplies by row weight.

    ``alpha`` is the kernel's positive/negative balance,
    ``alpha_t = alpha*y + (1-alpha)*(1-y)``. ``None`` leaves the terms
    unweighted, which is what a balanced multi-class label set wants; 0.25 is
    the detection setting for a rare positive class, and ``alpha=1`` is the
    degenerate end that zeroes every non-target class.

    ``gamma == 0.0`` with ``alpha=None`` dispatches to ``F.cross_entropy``:
    the unweighted sigmoid form at gamma 0 is a sum of binary cross-entropies
    over the class axis, not the categorical cross-entropy every cached cell
    was trained with. The switch is therefore exact at its disabled setting
    and the one-vs-all family otherwise.
    """
    if reduction not in ("none", "mean"):
        raise ValueError(f"reduction must be 'none' or 'mean', got {reduction!r}")
    if gamma == 0.0:
        return F.cross_entropy(logits, target, reduction=reduction)
    
    one_hot = F.one_hot(target, logits.size(-1)).to(logits.dtype)
    loss = sigmoid_focal_loss(
        logits, one_hot, alpha=-1.0 if alpha is None else float(alpha),
        gamma=gamma, reduction="none").sum(dim=-1)
    return loss.mean() if reduction == "mean" else loss


def _studentt_logprob(x: torch.Tensor, loc: torch.Tensor,
                      scale: torch.Tensor,
                      nu: torch.Tensor | float) -> torch.Tensor:
    """Diagonal multivariate Student-t log-density, summed over the last dim.

    ``nu`` is a scalar or a per-row ``(B,)`` tensor; a 1-D df is broadcast
    against the ``(B, D)`` value/loc/scale by adding the latent axis.
    """
    nu_t = (nu if torch.is_tensor(nu)
            else torch.as_tensor(nu, dtype=x.dtype, device=x.device))
    nu_t = nu_t.to(dtype=x.dtype, device=x.device)
    if nu_t.dim() == 1:
        nu_t = nu_t.unsqueeze(-1)
    z = (x - loc) / scale
    log_norm = (
        torch.lgamma((nu_t + 1.0) / 2.0)
        - torch.lgamma(nu_t / 2.0)
        - 0.5 * (torch.log(nu_t) + math.log(math.pi))
        - torch.log(scale)
    )
    half_nu_p1_div2 = (nu_t + 1.0) / 2.0
    log_kernel = -half_nu_p1_div2 * torch.log1p(z.square() / nu_t)
    return (log_norm + log_kernel).sum(dim=-1)


def _normal_logprob(x: torch.Tensor, loc: torch.Tensor,
                    scale: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log-density, summed over the last dim."""
    var = scale * scale
    log_p = -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(var)
        + torch.pow(x - loc, 2) / var
    )
    return log_p.sum(dim=-1)


class _LogVarHead(nn.Linear):
    """Linear log-variance head; the ``_LOGVAR_MAX`` clamp lives here.

    Every Gaussian factor in the model bounds its predicted log-variance, and
    the bound is a property of the head, not of the call site: keeping the
    ``clamp`` in ``forward`` stops the five heads drifting apart.
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return super().forward(h).clamp(max=_LOGVAR_MAX)


def _encoder_input(x: torch.Tensor) -> torch.Tensor:
    """The batch as the image encoder expects it.

    4-D image batches (NHWC) pass through unchanged; 3-D batches are flattened
    to features. Every entry point used to inline this test before deciding
    what to hand the encoder and the label head.
    """
    if x.dim() == 4:
        return x
    return x.flatten(start_dim=1) if x.dim() > 2 else x


def _softplus_inverse(y: float) -> float:
    """Pre-activation of a softplus that lands on ``y`` (``y > 0``)."""
    return math.log(math.expm1(y))


def _image_embed(encoder: nn.Module, x: torch.Tensor,
                 x_pool: torch.Tensor | None) -> torch.Tensor:
    """The deterministic (``mu``) image embedding of ``encoder``.

    ``x_pool`` is the pooled pre-head feature of the frozen pretrained trunk.
    LSNPC computes it once per call and hands the same tensor to both image
    encoders, so the trunk -- the expensive op -- runs once instead of once per
    encoder; the ``mu`` head that projects it to ``image_embed_dim`` stays
    private to each encoder. ``x_pool is None`` (no image data, or a trainable
    backbone whose two trunks diverge) falls back to the encoder's own forward.
    """
    if x_pool is None:
        mu, _ = encoder(x)
        return mu
    return encoder.mu(x_pool)


class ImagePass:
    """The pooled trunk features of one batch, computed at most once each.

    A training step reads the same batch through the image encoders several
    times -- the posterior pass, the clean-label pass and the label head -- and
    each read used to re-run a trunk. This holds each encoder's trunk output so
    the trunk runs once per batch and only the encoder's private ``mu`` head
    runs per read. `shared=True` (frozen trunks that are bit-identical) makes
    the label feature the posterior encoder's, which is what the frozen arm has
    always computed; a trainable arm keeps its two diverging trunks and pools
    each separately.

    Features are computed on first use so an entry point that reads only one
    encoder (``posterior_log_prob``, ``corrected_logits``) runs only that
    trunk.
    """

    def __init__(self, x: torch.Tensor, noisy_encoder: nn.Module,
                 label_encoder: nn.Module, shared: bool) -> None:
        self._x = x
        self._noisy_encoder = noisy_encoder
        self._label_encoder = label_encoder
        self._shared = bool(shared)
        self._noisy: torch.Tensor | None = None
        self._label: torch.Tensor | None = None

    @classmethod
    def shared(cls, pooled: torch.Tensor) -> "ImagePass":
        """Wrap a caller-supplied pooled feature that serves both encoders."""
        obj = cls.__new__(cls)
        obj._x = obj._noisy_encoder = obj._label_encoder = None
        obj._shared = True
        obj._noisy = obj._label = pooled
        return obj

    def noisy_feature(self) -> torch.Tensor:
        if self._noisy is None:
            self._noisy = self._noisy_encoder.backbone_feature(self._x)
            if self._shared:
                self._label = self._noisy
        return self._noisy

    def label_feature(self) -> torch.Tensor:
        if self._label is None:
            self._label = self._label_encoder.backbone_feature(self._x)
        return self._label


def _std_normal_logprob(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal N(0, I) log-density, summed over the last dim."""
    return (-0.5 * (math.log(2.0 * math.pi) + torch.pow(x, 2))).sum(dim=-1)


class NoisyEncoder(nn.Module):
    """q(ẑ | x, ŷ) = t_ν(μ_θ(x), diag(σ²_θ(x, ŷ))) with ν = ν_θ(x, ŷ).

    The image backbone named by ``encoder_backbone`` embeds ``x`` (image data);
    an MLP over ``[x_embed, ŷ]`` then produces the ẑ logvar and the per-sample
    df. Non-image input is used as-is. x is encoded here rather than read from a
    pre-encoded VAE latent, so the correction path stands on its own.

    The label is always in the trunk and never in the mean. ``μ`` is a Linear of
    ``x`` alone; the trunk -- and therefore both shape parameters, ``logvar``
    and ν -- reads ``[x, ŷ]``. The split is deliberate: the decoder reads the
    latent's *location*, so a label-dependent mean is a copy channel (the head
    can return ``ŷ`` without reading ``x``, which collapses the correction into
    an identity), whereas the shape parameters leave the location intact. See
    the implementation notes, §4.

    A ``RMSNorm`` (per-sample, no batch statistics) is applied to the trunk
    output before the logvar head for training stability.

    Scalar labels are one-hot encoded to ``n_classes`` dimensions.
    """

    def __init__(self, latent_dim: int, n_classes: int,
                 hidden_dim: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1,
                 image_data: bool = False, img_channels: int = 3,
                 img_size: int = 96, image_embed_dim: int = 128,
                 x_dim: int = 0, encoder_backbone: str = "conv",
                 freeze_backbone: bool = True,
                 label_dim: int | None = None):
        super().__init__()
        self.n_classes = n_classes
        # Width of the per-row label vector the trunk expects. Default is the
        # one-hot (n_classes); the shared label-embedding library (#4) passes
        # latent_dim and callers then hand in the embedded label.
        self.label_dim = n_classes if label_dim is None else int(label_dim)
        self.image_data = image_data

        self.image_encoder: nn.Module | None = None
        if image_data:
            self.image_encoder = build_image_encoder(
                encoder_backbone, image_embed_dim, img_channels, img_size,
                freeze_backbone=freeze_backbone)
            x_feat_dim = image_embed_dim
        else:
            x_feat_dim = x_dim

        # Trunk input dim = feature dimension + label embedding dimension.
        in_dim = x_feat_dim + self.label_dim
        dims = [in_dim] + [hidden_dim] * n_blocks
        self.trunk = build_mlp(dims, dropout=dropout)
        self.norm = nn.RMSNorm(hidden_dim)
        # μ reads x alone. The label conditions the trunk, and with it both
        # shape parameters; it must not reach the location, which is the one
        # parameter the decoder reads for the copy
        # (the implementation notes, §4).
        self.mu = nn.Linear(x_feat_dim, latent_dim)
        self.logvar = _LogVarHead(hidden_dim, latent_dim)
        # ν(x, ŷ): one df per row, off the same trunk. The bias starts the head
        # at the df the posterior used to hold fixed, so a fresh model opens on
        # the previous Student-t.
        self.nu = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.nu.weight)
        nn.init.constant_(self.nu.bias, _softplus_inverse(_NU_INIT - _NU_MIN))

    def nu_from_state(self, h: torch.Tensor) -> torch.Tensor:
        """ν(x, ŷ), shape ``(B,)``: the trunk state's per-row df."""
        nu = _NU_MIN + F.softplus(self.nu(h))
        return nu.clamp(max=_NU_MAX).squeeze(-1)

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor,
                img_pass: "ImagePass | None" = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.image_encoder is not None:
            x = _image_embed(
                self.image_encoder, x,
                None if img_pass is None else img_pass.noisy_feature())
        elif x.dim() > 2:
            x = x.flatten(start_dim=1)

        if y_hat.dim() == 1:
            y_hat = class_to_onehot(y_hat, self.n_classes, device=x.device)
        y_hat = y_hat.to(x.dtype)
        h = self.trunk(torch.cat([x, y_hat], dim=-1))
        h = self.norm(h)  # RMSNorm: per-sample, safe at any batch size
        return self.mu(x), self.logvar(h), self.nu_from_state(h)


class CorrectionEncoder(nn.Module):
    """q(z | ẑ, z_vae[, cond]) = N(μ_κ(ẑ, z_vae), diag(σ²_κ(ẑ))).

    Gated interpolation (v1 fix, ported to v2): μ is a convex combination of
    the noisy latent ẑ and the clean VAE encoding z_vae,
        μ = gate ⊙ ẑ + (1 − gate) ⊙ z_vae,   gate = sigmoid(gate_net([ẑ, z_vae]))
    so the corrected latent stays bounded by [ẑ, z_vae].  The unconstrained
    MLP version projected the correction off the VAE manifold and destroyed
    the image content of the corrected latent (cos(z_a, z_vae) ≈ 0.02 → flat SFs).
    The gate starts at ~0.5 (zero-initialised last bias) and learns which
    dims to trust from ẑ vs z_vae.  Variance comes from ẑ only (original).

    ``cond`` (a caller-built vector: the image embedding, optionally with the
    one-hot noisy label) enters the *gate*, never the mean blend, so the
    convex-combination property above holds whatever the conditioning says.

    The second endpoint is the caller's choice of blend target: the VAE latent
    ``z_vae`` (the ``q(z | ẑ, x)`` of the paper) or, for the ``yhat`` arm, a
    learnable per-class embedding of the noisy label (``q(z | ẑ, ŷ)``).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_blocks: int = 2, dropout: float = 0.1,
                 cond_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = int(cond_dim)
        gate_dims = ([2 * latent_dim + self.cond_dim]
                     + [hidden_dim] * n_blocks + [latent_dim])
        self.gate_net = build_mlp(gate_dims, dropout=dropout)
        gate_last = self.gate_net[-1]
        assert isinstance(gate_last, nn.Linear)
        nn.init.zeros_(gate_last.bias)  # sigmoid(0) ≈ 0.5 at init
        var_dims = [latent_dim] + [hidden_dim] * n_blocks
        self.var_trunk = build_mlp(var_dims, dropout=dropout)
        self.logvar = _LogVarHead(hidden_dim, latent_dim)

    def forward(self, z_hat: torch.Tensor, z_vae: torch.Tensor,
                cond: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [z_hat, z_vae]
        if self.cond_dim:
            if cond is None:
                raise ValueError(
                    "this CorrectionEncoder is conditioned "
                    f"(cond_dim={self.cond_dim}); pass cond")
            parts.append(cond)
        gate = torch.sigmoid(self.gate_net(torch.cat(parts, dim=-1)))
        mu = gate * z_hat + (1.0 - gate) * z_vae
        h = self.var_trunk(z_hat)
        logvar = self.logvar(h)
        return mu, logvar


class _MLPGaussianHead(nn.Module):
    """h = trunk(z); (μ, log σ²) = heads(h) — a diagonal Gaussian over z.

    Two instances carry opposite roles and identical shape: the generative
    shift ``p(ẑ | z) = t_ν0(μ_ψ(z), σ²_ψ(z))`` and the inference-only data
    decoder ``p(x | z) = N(μ_φ(z), σ²_φ(z))``, which no loss term reads
    (``LSNPC.decode`` is its only caller). They differ only in output width.
    """

    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        dims = [latent_dim] + [hidden_dim] * n_blocks
        self.trunk = build_mlp(dims, dropout=dropout)
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.logvar = _LogVarHead(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        return self.mu(h), self.logvar(h)


class GatedShiftDecoder(nn.Module):
    """Gated residual shift decoder — refines the corrected latent before
    the data decoder is applied.

    Pattern adapted from ``MlcDecoderZ`` in the reference ``nlc_vae.py``:
    a single MLP trunk projects to ``2 * latent_dim`` channels which are
    split into a sigmoid gate ``g`` and a residual delta ``d_z_hat``; the
    output is the convex blend

        z_gated = g * z_hat + (1 - g) * d_z_hat

    so that ``g`` smoothly interpolates between the identity (``g = 0``,
    pass-through) and a learned residual correction (``g = 1``).  The
    gate's sigmoid makes the mapping Lipschitz-bounded and stable across
    training.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        dims = [latent_dim] + [hidden_dim] * (n_layers - 1) + [latent_dim * 2]
        self.shift_mlp = build_mlp(dims, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.shift_mlp(z)
        gate, d_z_hat = out.chunk(2, dim=-1)
        g = torch.sigmoid(gate)
        return g * z + (1.0 - g) * d_z_hat


#: Valid ablation modes for the label head g(·).
class _LabelHead(nn.Module):
    """g_φ(x, z) → class logits (shared corrected predictor). NOTE: the paper's g_φ is the
    normalised class distribution, i.e. softmax of these logits; the softmax is applied by
    the callers (corrected_conditioning, decode) and never twice.

    For image data (``image_data=True``) the image backbone's deterministic
    ``mu`` is the ``image_embed_dim`` embedding; it is concatenated with ``z``
    and fed into a small MLP head.
    """

    def __init__(self, x_dim: int, latent_dim: int, n_classes: int,
                 hidden_dim: int = 128, n_blocks: int = 2, dropout: float = 0.1,
                 image_data: bool = False, img_channels: int = 3,
                 img_size: int = 96, image_embed_dim: int = 128,
                 encoder_backbone: str = "conv",
                 freeze_backbone: bool = True, head_type: str = "concat",
                 ):
        super().__init__()
        self.image_data = image_data

        self.image_encoder: nn.Module | None = None
        if image_data:
            self.image_encoder = build_image_encoder(
                encoder_backbone, image_embed_dim, img_channels, img_size,
                freeze_backbone=freeze_backbone)
            x_feat_dim = image_embed_dim
        else:
            x_feat_dim = x_dim

        self.head_type = str(head_type)
        if self.head_type not in ("concat", "gate"):
            raise ValueError(f"unsupported label-head type {self.head_type!r}; "
                             "expected 'concat' or 'gate'")
        # ``concat`` (original) reads both inputs with one trunk, which may
        # ignore the image entirely; ``gate`` runs one trunk per input and mixes
        # their logits, ``λ·f_x(x_feat) + (1-λ)·f_z(z)``, so reliance on each
        # pathway becomes observable in ``last_gate`` (λ -> 0 = the label copy).
        # It gives up x⊗z interactions, which is the trade.
        if self.head_type == "gate":
            self.x_trunk = build_mlp([x_feat_dim, hidden_dim, hidden_dim],
                                     dropout=dropout)
            self.z_trunk = build_mlp([latent_dim, hidden_dim, hidden_dim],
                                     dropout=dropout)
            self.x_out = nn.Linear(hidden_dim, n_classes)
            self.z_out = nn.Linear(hidden_dim, n_classes)
            self.gate_x = nn.Linear(x_feat_dim, 1)
            self.gate_z = nn.Linear(latent_dim, 1)
            self.last_gate: float | None = None
        else:
            dims = ([x_feat_dim + latent_dim] + [hidden_dim] * n_blocks
                    + [n_classes])
            self.norm = nn.BatchNorm1d(dims[0])
            self.net = build_mlp(dims, dropout=dropout)

    def embed_x(self, x: torch.Tensor,
                img_pass: "ImagePass | None" = None) -> torch.Tensor:
        """Encode the input ``x`` to the (B, image_embed_dim) feature tensor
        consumed by ``head``.

        For image data the deterministic ``mu`` of the image encoder is
        returned; otherwise ``x`` is flattened (or returned unchanged).

        Deterministic in the batch; safe to call once per batch and reuse
        the result across multiple ``head`` calls (e.g. inside a K-loop).
        ``img_pass`` carries this batch's pooled trunk features
        (see ``ImagePass``); the head reads the label encoder's.
        """
        if self.image_encoder is not None:
            return _image_embed(
                self.image_encoder, x,
                None if img_pass is None else img_pass.label_feature())
        return x.flatten(start_dim=1) if x.dim() > 2 else x

    def head(self, x_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Run only the MLP head over a pre-encoded ``x_feat`` and latent
        ``z``. Pairs with ``embed_x`` for K-loop callers that want to
        encode the input exactly once per batch."""
        if self.head_type == "gate":
            gate = torch.sigmoid(self.gate_x(x_feat) + self.gate_z(z))
            logits = (gate * self.x_out(self.x_trunk(x_feat))
                      + (1.0 - gate) * self.z_out(self.z_trunk(z)))
            # Diagnostic readout, deliberately a plain float: no graph, no
            # state_dict entry, no effect on the frozen bundles.
            self.last_gate = float(gate.detach().mean())
            return logits
        return self.net(self.norm(torch.cat([x_feat, z], dim=-1)))

    def forward(self, x: torch.Tensor, z: torch.Tensor,
                img_pass: "ImagePass | None" = None) -> torch.Tensor:
        """Single-call wrapper for callers that don't need to share an
        embedding across multiple ``z`` values. Training-time K-loop
        callers should use ``embed_x`` once and ``head`` per ``z``."""
        return self.head(self.embed_x(x, img_pass), z)


class LSNPC(nn.Module):
    """Combined LSNPC module: two inference encoders + generative shift +
    shared decoder g_φ (label predictor head + inference-only data decoder).

    The frozen VAE encoder ``VAE_enc`` supplies the deterministic latent
    ``z_vae`` that conditions the NoisyEncoder; it is passed in at call time
    (not owned by this module) so the same VAE trained in the pipeline can be
    reused.

    The label head outputs ``n_classes`` logits, losses use cross-entropy,
    and ``corrected_conditioning`` returns a softmax distribution.
    """

    def __init__(self, x_dim: int, latent_dim: int, n_classes: int = 2,
                 hidden_dim: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1,
                 nu0: float = 2.0,
                 focal_gamma: float = 0.0,
                 focal_alpha: float | None = None,
                 image_data: bool = False,
                 img_channels: int = 3,
                 img_size: int = 96,
                 image_embed_dim: int = 128,
                 beta: float = 1.0,
                 encoder_backbone: str = "conv",
                 freeze_backbone: bool = True,
                 head_type: str = "concat",
                 correction_cond: str = "none",
                 shared_yhat_embed: bool = False,
                 correction_input: str = "gated"):
        super().__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.nu0 = float(nu0)                  # prior (generative) df
        self.out_dim: int = n_classes
        self.focal_gamma = float(focal_gamma)  # 0.0 = plain cross-entropy
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)
        self.image_data = bool(image_data)
        self.beta = float(beta)   # KL coefficient of the closed-form correction objective (paper eq:correction_loss); replaces the former kl_weight
        # #4: one shared learnable label-embedding library, used by BOTH the
        # posterior's label slot and the blend target, so every place the
        # noisy label enters reads the same learned per-class embedding.
        self._shared_yhat = bool(shared_yhat_embed)
        # Correction-map input: "gated" is the original convex blend of the
        # noisy latent toward the blend target; "concat" feeds cat([ẑ, x_feat]) to a
        # trunk, so the corrected latent is a learned function of ẑ AND x.
        self.correction_input = str(correction_input)
        if self.correction_input not in ("gated", "concat"):
            raise ValueError(
                "correction_input must be 'gated' or 'concat', got "
                f"{self.correction_input!r}")
        # What the correction map q(z | ẑ, ·) reads:
        #   "none"   -- the VAE latent z_vae only (the paper's q(z | ẑ, x));
        #   "x"      -- z_vae, with the image embedding added to the gate;
        #   "x_yhat" -- z_vae, with the image embedding and the one-hot noisy
        #               label added to the gate (q(z | ẑ, x, ŷ));
        #   "yhat"   -- no x at all: the VAE latent is replaced by a learnable
        #               per-class embedding of the noisy label (q(z | ẑ, ŷ)).
        self.correction_cond = str(correction_cond)
        if self.correction_cond not in ("none", "x", "x_yhat", "yhat"):
            raise ValueError(
                "correction_cond must be 'none', 'x', 'x_yhat' or 'yhat', got "
                f"{self.correction_cond!r}")
        self._cond_x = self.correction_cond in ("x", "x_yhat")
        self._cond_yhat = self.correction_cond == "x_yhat"
        self._blend_target_from_label = self.correction_cond == "yhat"
        self._current_epoch = 0
        if self.nu0 <= 0.0:
            raise ValueError("LSNPC prior degrees of freedom must be positive.")

        # Inference (variational) heads. NoisyEncoder's trunk input dim must
        # match the ``z_vae`` shape callers pass to forward()/iw_log_weights(),
        # which is ``LSNPC.latent_dim``.
        self.noisy_encoder = NoisyEncoder(
            latent_dim, self.out_dim, hidden_dim, n_blocks, dropout,
            image_data=image_data,
            img_channels=img_channels, img_size=img_size,
            image_embed_dim=image_embed_dim,
            x_dim=x_dim, encoder_backbone=encoder_backbone,
            freeze_backbone=freeze_backbone,
            label_dim=(latent_dim if self._shared_yhat else None))
        cond_dim = 0
        if self._cond_x:
            cond_dim = image_embed_dim if image_data else x_dim
            if self._cond_yhat:
                cond_dim += n_classes
        self.correction_encoder = CorrectionEncoder(
            latent_dim, hidden_dim, n_blocks, dropout, cond_dim=cond_dim)
        _x_feat_dim = image_embed_dim if image_data else x_dim
        self.correction_concat = (build_mlp(
            [latent_dim + _x_feat_dim] + [hidden_dim] * n_blocks, dropout=dropout)
            if self.correction_input == "concat" else None)
        self.correction_concat_mu = (nn.Linear(hidden_dim, latent_dim)
                                     if self.correction_input == "concat" else None)
        self.correction_concat_logvar = (_LogVarHead(hidden_dim, latent_dim)
                                         if self.correction_input == "concat" else None)
        # Blend target of the "yhat" arm: a learnable latent per noisy class,
        # standing in for the VAE latent the other arms blend against.
        self.yhat_blend_target = (nn.Linear(n_classes, latent_dim)
                            if self._blend_target_from_label else None)
        self.shared_yhat_embed = (nn.Embedding(n_classes, latent_dim)
                                  if self._shared_yhat else None)
        # The shared library feeds the POSTERIOR's label slot. It reaches the
        # blend target only when that blend target is active (correction_cond
        # 'yhat'), so 'variance' + shared keeps the VAE latent on z_vae.
        # Generative heads (shared decoder g_φ)
        self.shift_model = _MLPGaussianHead(
            latent_dim, latent_dim, hidden_dim, n_blocks, dropout)
        # Gated residual shift: refines the corrected latent before the
        # data decoder runs (MlcDecoderZ pattern).
        self.gated_decoder = GatedShiftDecoder(
            latent_dim, hidden_dim, n_blocks, dropout)

        self.data_decoder = _MLPGaussianHead(
            latent_dim, x_dim, hidden_dim, n_blocks, dropout)
        self.label_head = _LabelHead(
            x_dim, latent_dim, self.out_dim, hidden_dim, n_blocks, dropout,
            image_data=image_data, img_channels=img_channels,
            img_size=img_size, image_embed_dim=image_embed_dim,
            encoder_backbone=encoder_backbone,
            freeze_backbone=freeze_backbone, head_type=head_type)

        # Set by the trainer: no loss term reads the data decoder, so its
        # parameters are frozen out of training.
        self.phase1_wo_decoder = False

        # Both encoders own a copy of the same pretrained trunk. Two frozen
        # copies are bit-identical, so one forward can feed both mu heads; a
        # trainable copy diverges and is never shared.
        self._share_backbone_eligible = bool(
            image_data and freeze_backbone
            and encoder_backbone in FROZEN_TRUNK_BACKBONES)
        self.refresh_shared_trunk()

    def correction_cond_vector(self, x_feat: torch.Tensor,
                               y_hat: torch.Tensor) -> torch.Tensor | None:
        """The ``cond`` argument of ``q(z | ẑ, ·)``.

        ``None`` for the arms that read no x (``none``, ``yhat``). Otherwise
        ``x_feat`` is the image embedding the label head already computes (no
        extra trunk pass), with the one-hot noisy label appended in full.
        """
        if not self._cond_x:
            return None
        if not self._cond_yhat:
            return x_feat
        if y_hat.dim() == 1:
            y_hat = class_to_onehot(y_hat, self.n_classes, device=x_feat.device)
        return torch.cat([x_feat, y_hat.to(x_feat.dtype)], dim=-1)

    def _correction_map(self, z_hat, x_feat, z_vae, y_hat, cond):
        """The correction map q(z | ẑ, ·) -> (mu, logvar).

        'gated' keeps the original convex blend of ẑ toward the blend target;
        'concat' runs cat([ẑ, x_feat]) through a trunk, so the corrected
        latent is a learned function of the noisy latent and the input.
        """
        if self.correction_input == "concat":
            h = self.correction_concat(torch.cat([z_hat, x_feat], dim=-1))
            mu = self.correction_concat_mu(h)
            logvar = self.correction_concat_logvar(h)
            return mu, logvar
        return self.correction_encoder(
            z_hat, self.blend_target(z_vae, y_hat), cond)

    def blend_target(self, z_vae: torch.Tensor,
                          y_hat: torch.Tensor) -> torch.Tensor:
        """The second endpoint of the correction map's mean blend.

        ``z_vae`` (the frozen VAE latent of x) in every arm but ``yhat``,
        which replaces the image with the weighted label embedding and so
        maps ``q(z | ẑ, ŷ)`` instead of ``q(z | ẑ, x)``.
        """
        if not self._blend_target_from_label:
            return z_vae
        if self._shared_yhat:
            idx = y_hat if y_hat.dim() == 1 else y_hat.argmax(dim=-1)
            return self.shared_yhat_embed(idx.long())
        if y_hat.dim() == 1:
            y_hat = class_to_onehot(y_hat, self.n_classes, device=z_vae.device)
        w = self.yhat_blend_target.weight
        return self.yhat_blend_target(y_hat.to(w.dtype))

    def _posterior_label(self, y: torch.Tensor) -> torch.Tensor:
        """Label handed to the posterior encoder. Identity unless the shared
        label library is on, in which case the input is the shared per-class
        embedding instead of the raw one-hot."""
        if not self._shared_yhat:
            return y
        idx = y if y.dim() == 1 else y.argmax(dim=-1)
        return self.shared_yhat_embed(idx.long())

    def refresh_shared_trunk(self) -> bool:
        """Re-decide ``share_backbone_pass`` from the current trunk weights.

        Called at construction and again after a checkpoint load: a bundle can
        carry two trunk copies that were fine-tuned apart, and sharing one
        forward between them would then be silently wrong. Returns the flag.
        """
        share = self._share_backbone_eligible and self._trunks_identical()
        if self._share_backbone_eligible and not share:
            log.warning(
                "the two frozen image-encoder trunks differ (randomly "
                "initialised, or a bundle whose copies were fine-tuned "
                "apart); each encoder will run its own trunk forward.")
        self.share_backbone_pass = share
        return share

    def _trunks_identical(self) -> bool:
        """Whether both image encoders carry identical trunk weights.

        Reusing one trunk forward for both is only valid when it cannot change
        the result. Any difference (a randomly initialised backbone, or a
        loaded bundle whose two copies were fine-tuned apart) turns sharing off
        instead of silently feeding the label head the posterior encoder's
        features. The ``mu``/``logvar`` projection heads are private to each
        encoder and deliberately excluded.
        """
        a = self.noisy_encoder.image_encoder
        b = self.label_head.image_encoder
        if a is b:
            return True
        sa, sb = a.state_dict(), b.state_dict()
        if sa.keys() != sb.keys():
            return False
        return all(torch.equal(sa[k], sb[k]) for k in sa
                   if not k.startswith(("mu.", "logvar.")))

    def image_pool(self, x: torch.Tensor,
                   x_pool: torch.Tensor | None = None) -> torch.Tensor | None:
        """Pooled frozen-trunk feature handed to both image encoders.

        Computed here when the caller passes ``None``, so a call that needs the
        image embedding for several modules (``compute_loss``) or several times
        (``iw_log_weights``) runs the trunk once. Callers that re-encode the
        same images repeatedly (the protocol scorer's K perturbation draws)
        pass a precomputed tensor instead. ``None`` means "not shared": no
        image data, or a trainable backbone, and each module then pools its
        own trunk.
        """
        if x_pool is not None:
            if not self.share_backbone_pass:
                raise ValueError(
                    "x_pool requires a model that shares one frozen trunk "
                    "between its image encoders; this one does not, so the "
                    "pooled features would not be the ones its modules use.")
            return x_pool
        if not self.share_backbone_pass:
            return None
        return self.noisy_encoder.image_encoder.backbone_feature(x)

    def image_pass(self, x: torch.Tensor,
                   x_pool: "torch.Tensor | ImagePass | None" = None,
                   ) -> "ImagePass | None":
        """This batch's pooled trunk features, one per image encoder.

        ``None`` for non-image models. A caller-supplied ``x_pool`` is used
        as-is when it is already an ``ImagePass`` (a trainable model's two
        per-module features); a bare tensor is one pooled feature that serves
        both encoders and therefore requires ``share_backbone_pass``.
        Otherwise each encoder's trunk is pooled on first use, so a step that
        reads the encoders several times runs each trunk once.
        """
        if isinstance(x_pool, ImagePass):
            return x_pool
        if x_pool is not None:
            if not self.share_backbone_pass:
                raise ValueError(
                    "a shared x_pool requires a model with one frozen trunk "
                    "behind both image encoders; this one does not have one, "
                    "so those features would not be the ones its modules use. "
                    "Pass an ImagePass with the per-encoder features instead.")
            return ImagePass.shared(x_pool)
        if not self.image_data:
            return None
        return ImagePass(x, self.noisy_encoder.image_encoder,
                         self.label_head.image_encoder,
                         shared=self.share_backbone_pass)

    def set_current_epoch(self, epoch: int):
        """Forward the current training epoch (reserved; no KL warm-up schedule
        remains — the KL coefficient is the constant ``beta`` of eq:correction_loss)."""
        self._current_epoch = epoch

    # ── Forward (IW path, unchanged API) ──────────────────────────────

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor,
                z_vae: torch.Tensor | None = None,
                K: int = 1,
                x_pool: "torch.Tensor | ImagePass | None" = None) -> dict:
        """Forward pass: sample K (ẑ, z) pairs from the variational path.

        Returns:
            dict with keys: z_samples (K, B, d), log_w (B, K),
            z_mean (B, d), corrected_logits (B, n_classes).
        """
        B = x.size(0)
        if z_vae is None:
            z_vae = torch.zeros(B, self.latent_dim, device=x.device,
                                dtype=x.dtype)
        img_pass = self.image_pass(x, x_pool)
        log_w, z_samples = self.iw_log_weights(x, z_vae, y_hat, K,
                                              x_pool=img_pass)

        # Corrected predictor: use IW-mean of z
        z_mean = self.importance_weighted_mean(log_w, z_samples)

        corrected_logits = self.label_head(x, z_mean, img_pass=img_pass)

        return {
            'z_samples': z_samples,
            'log_w': log_w,
            'z_mean': z_mean,
            'corrected_logits': corrected_logits,
        }

    # ── Reparameterised sampling ──────────────────────────────────────
    @staticmethod
    def importance_weighted_mean(
        log_w: torch.Tensor, z_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Return the self-normalised importance-weighted latent mean.

        ``log_w`` has shape ``(B, K)`` and ``z_samples`` has shape
        ``(K, B, D)``. Keeping this operation shared makes clean-set-loss
        training and inference use the same corrected-latent estimator.
        """
        if log_w.ndim != 2 or z_samples.ndim != 3:
            raise ValueError("Expected log_w=(B,K) and z_samples=(K,B,D).")
        if log_w.shape[0] != z_samples.shape[1] or log_w.shape[1] != z_samples.shape[0]:
            raise ValueError("Importance weights and latent samples have incompatible shapes.")
        weights = torch.softmax(log_w, dim=-1)
        return (weights.unsqueeze(-1) * z_samples.permute(1, 0, 2)).sum(dim=1)

    # Reparameterised sampling helpers.
    @staticmethod
    def _logvar_to_scale(logvar: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * logvar).clamp(min=_STD_MIN)

    def _sample_studentt(self, mu: torch.Tensor, logvar: torch.Tensor,
                         nu: torch.Tensor) -> torch.Tensor:
        """Reparameterised Student-t draw via ``D.StudentT.rsample``.

        Student-t sampling has no autocast policy and degrades under bf16
        (8-bit mantissa), so the draw is computed at fp32 on CPU
        (graph-preserving: ``.float().cpu()`` keeps autograd) and moved
        back to the original device/dtype (bf16 under autocast) before it
        re-enters the autograd graph.

        ``nu`` is the posterior's per-row df; it is broadcast over the latent
        axis to line up with the ``(B, D)`` location and scale.
        """
        scale = self._logvar_to_scale(logvar)
        mu_f = mu.float().cpu()
        scale_f = scale.float().cpu()
        T = D.StudentT(df=nu.float().cpu().reshape(-1, 1),
                       loc=mu_f, scale=scale_f)
        return T.rsample().to(mu.device).to(mu.dtype)

    @torch.no_grad()
    def posterior_log_prob(self, x: torch.Tensor, y_hat: torch.Tensor,
                           value: torch.Tensor,
                           x_pool: "torch.Tensor | ImagePass | None" = None,
                           ) -> torch.Tensor:
        """Posterior density of ``value`` under q(ẑ|x,ŷ) = T_ν(μθ, σθ).

        The corrected code is scored against the encoder posterior rather
        than the standard prior: plausibility of an edit under the learned
        model (mirrors the ``logq`` term of the correction objective).
        ``x`` may be 4-D image or flat features; y_hat are the label
        indices. Returns per-row log-density summed over latent dims.
        """
        x_enc = _encoder_input(x)
        mu, logvar, nu = self.noisy_encoder(
            x_enc, self._posterior_label(y_hat),
            img_pass=self.image_pass(x_enc, x_pool))
        scale = self._logvar_to_scale(logvar)
        return self._studentt_log_prob(value, mu, scale, nu=nu).sum(dim=-1)

    def _studentt_log_prob(self, value: torch.Tensor,
                           loc: torch.Tensor | None = None,
                           scale: torch.Tensor | None = None,
                           nu: torch.Tensor | float | None = None
                           ) -> torch.Tensor:
        """Per-element Student-t log-prob at fp32 (CPU) — the ``lgamma`` /
        ``log1p`` kernel loses precision in bf16 and ``D.StudentT.log_prob``
        has no autocast policy. Returns a tensor on ``value``'s device/dtype.

        ``loc``/``scale`` default to the standard Student-t. ``nu`` is the
        df: a scalar (the prior's ν0) or the posterior's per-row ``(B,)``
        tensor, broadcast over the latent axis.
        """
        if torch.is_tensor(nu) and nu.dim() == 1:
            nu = nu.float().cpu().reshape(-1, 1)
        value_f = value.float().cpu()
        if loc is None:
            T = D.StudentT(df=nu)
        else:
            T = D.StudentT(
                df=nu, loc=loc.float().cpu(),
                # `scale` is optional; StudentT defaults it to 1.0.
                scale=None if scale is None else scale.float().cpu())
        return T.log_prob(value_f).to(value.device).to(value.dtype)

    def _sample_normal(self, mu: torch.Tensor, logvar: torch.Tensor
                       ) -> torch.Tensor:
        scale = self._logvar_to_scale(logvar)
        return mu + scale * torch.randn_like(mu)

    # ── Corrected predictor h̃(x, ŷ) = g_φ(x, z) ──────────────────────

    def corrected_logits(self, x: torch.Tensor, z: torch.Tensor,
                         x_pool: "torch.Tensor | ImagePass | None" = None,
                         ) -> torch.Tensor:
        """h̃ logits: the corrected label prediction from the LSNPC path."""
        return self.label_head(x, z, img_pass=self.image_pass(x, x_pool))

    # No @torch.no_grad() here: callers own the grad context.
    def decode(self, z: torch.Tensor, c: torch.Tensor | None = None
               ) -> torch.Tensor:
        """Decode the corrected latent ``z`` through the data decoder.

        Inference-only: the recourse path, not the loss. ``c`` is accepted and
        ignored (the decoder is unconditional). ``z`` is first refined by the
        gated residual shift decoder. Returns the flat ``(B, D)`` decoder mean.
        """
        mu, _ = self.data_decoder(self.gated_decoder(z))
        return mu

    # ── IW-weighted clean-set loss (paper Eq. 6, K-sample expectation) ───

    def clean_set_loss(
        self, x: torch.Tensor, y_hat: torch.Tensor, y_clean: torch.Tensor,
        z_vae: torch.Tensor | None = None, K: int = 5,
        beta: float = 1.0,
        row_weight: torch.Tensor | None = None,
        x_pool: "torch.Tensor | ImagePass | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IW-weighted clean-set loss (paper Eq. 6).

        ``beta`` is the fractional posterior exponent (paper's $\beta$): it
        tempers the likelihood terms inside the importance weights
        (default 1.0 = standard posterior).

        IW-weighted cross-entropy
            L_clean = sum_k w_tilde_k * CE(g_phi(x, z^(k)), y*)
        where ``y*`` is the clean supervision target (scalar labels).

        ``w̃_k = softmax(log w_k)`` are the normalised importance weights of
        the *unsupervised* posterior. Gradients flow through both the
        per-sample loss and the label head (the weights are detached so the
        clean-set term supervises g_φ rather than reshaping the posterior).

        ``row_weight`` (B,) scales each row's contribution: a per-row
        confidence on the clean supervision, used when part of the clean set
        carries corrected labels rather than known-clean ones. Uniform (1.0)
        by default, which reproduces the unweighted loss exactly.

        Returns ``(loss, w_tilde)`` where ``w_tilde`` is (B, K).
        """
        B = x.size(0)
        if z_vae is None:
            z_vae = torch.zeros(B, self.latent_dim, device=x.device,
                                dtype=x.dtype)
        x_enc = _encoder_input(x)
        img_pass = self.image_pass(x_enc, x_pool)
        log_w, z_samples = self.iw_log_weights(
            x, z_vae, y_hat, K, y_clean=None, beta=beta, x_pool=img_pass)
        # Normalised importance weights (detached — they weight, not reshape).
        w_tilde = torch.softmax(log_w.detach(), dim=-1)          # (B, K)
        # Hoist label_head.embed_x once: the input encoder is the heavy
        # op on image backbones and is invariant over the K samples.
        x_feat = self.label_head.embed_x(x_enc, img_pass=img_pass)
        loss = x.new_zeros(())
        for k in range(K):
            pred_k = self.label_head.head(x_feat, z_samples[k])
            sample_loss = focal_cross_entropy(
                pred_k, y_clean, gamma=self.focal_gamma,
                alpha=self.focal_alpha, reduction="none")  # (B,)
            w_row = (1.0 if row_weight is None
                     else row_weight.reshape(-1).to(sample_loss.dtype))
            loss = loss + (w_tilde[:, k] * sample_loss * w_row).sum()
        loss = loss / B
        return loss, w_tilde

    # ── Closed-form CorrectionLoss (training objective, v2) ───────────


    def compute_loss(
        self, x: torch.Tensor, y_hat: torch.Tensor,
        y_clean: torch.Tensor | None = None,
        beta: float = 1.0, eta: float = 0.1,
        z_vae: torch.Tensor | None = None,
        K_clean_set: int = 5,
        clean_row_weight: torch.Tensor | None = None,
        x_pool: "torch.Tensor | ImagePass | None" = None,
    ) -> dict:
        """Closed-form CorrectionLoss surrogate (reference ``CorrectionLoss``).

        Two-phase schedule (paper sec.4): with probability ``eta`` (and only when
        ``y_clean`` is provided) the second encoder pass conditions on the
        *clean* label (semi-supervised clean-set pass); otherwise it maps the
        noisy latent ẑ → z through the CorrectionEncoder (unsupervised pass).

        Loss (paper eq:correction_loss):
            recon       = CE(g_φ(x, ẑ), ŷ)                     (unweighted)
            clean_set   = Σ_k w̃_k · CE(g_φ(x, z^(k)), y*)      [if y_clean]
            kl_zhat     = -E[exp(logp - logq) - 1 - (logp - logq)]   (Student-t)
            kl_z        = -0.5·E[1 + logvar - μ² - exp(logvar)]        (Gaussian)
            loss        = recon + clean_set + β·(kl_zhat + kl_z)

        ``beta`` (paper's $\\beta$) is the KL coefficient: it replaces the
        former ``kl_weight``/``w_KL`` machinery.  The clean-set loss's IW
        weights are untempered (``beta=1.0`` in the clean-set-loss call below).

        The clean-set term (v3) is the **IW-weighted** cross-entropy over
        ``K_clean_set`` posterior samples (paper Eq. 6), replacing the
        previous single-sample CE on the IW-mean latent.

        where logp = t_{ν0}(ẑ - μ_ψ(z)), logq = t_ν(ẑ; μ_θ, σ_θ).

        Returns:
            dict with 'loss', 'recon', 'kl_zhat', 'kl_z' (+ 'recon_yhat',
            'clean_set_loss', 'recon_y' components).
        """
        # ── Responsibly reshape image-data inputs before flattening ──
        # Encoder input: the 4D image (image data) or the flat feature vector.
        x_enc = _encoder_input(x)
        # One trunk forward per encoder feeds every image-encoder read below
        # (posterior, clean-set pass, label head).
        img_pass = self.image_pass(x_enc, x_pool)

        # q(ẑ | x, ŷ)  — Student-t posterior over the noisy latent.
        mu_theta, logvar_theta, nu_theta = self.noisy_encoder(
            x_enc, self._posterior_label(y_hat), img_pass=img_pass)
        std_theta = self._logvar_to_scale(logvar_theta)
        z_hat = self._sample_studentt(mu_theta, logvar_theta, nu_theta)

        # Hoist the image embedding: the trunk is the heavy op, and both the
        # correction map's conditioning and the label head read it below.
        x_feat = self.label_head.embed_x(x_enc, img_pass=img_pass)
        cond = self.correction_cond_vector(x_feat, y_hat)

        # Second encoder pass: semi-supervised (clean) with prob eta, else
        # the unsupervised correction map q(z | ẑ, z_vae, ·).
        # Use stdlib random — avoids a GPU kernel + host sync per batch.
        use_clean = (y_clean is not None) and (random.random() < eta)
        if use_clean:
            mu_kappa, logvar_kappa, _ = self.noisy_encoder(
                x_enc, self._posterior_label(y_clean), img_pass=img_pass)
        else:
            mu_kappa, logvar_kappa = self._correction_map(
                z_hat, x_feat, z_vae, y_hat, cond)
        z = self._sample_normal(mu_kappa, logvar_kappa)

        # Generative location of ẑ given z (μ_ψ(z)) for the KL surrogate.
        z_dec_mu, _ = self.shift_model(z)

        # Noisy-label reconstruction (``x_feat`` hoisted above).
        recon_yhat_pred = self.label_head.head(x_feat, z_hat)
        label_loss = focal_cross_entropy(recon_yhat_pred, y_hat,
                                         gamma=self.focal_gamma,
                                         alpha=self.focal_alpha)

        # Objective: the label evidence of eq. 3 plus the two divergences.
        # The data-reconstruction term was removed (S3l), so the decoder is
        # never read here.
        recon_loss = label_loss

        # IW-weighted clean-set CE (paper Eq. 6) — K-sample expectation.
        clean_set = torch.zeros((), device=x.device, dtype=recon_loss.dtype)
        recon_y = torch.zeros((), device=x.device, dtype=recon_loss.dtype)
        if y_clean is not None:
            # Objective call: beta lives on the KL terms (paper eq:correction_loss),
            # so the clean-set loss's IW weights use the untempered posterior (beta=1.0).
            clean_set, _w = self.clean_set_loss(
                x_enc, y_hat, y_clean, z_vae=z_vae, K=K_clean_set,
                beta=1.0, row_weight=clean_row_weight, x_pool=img_pass)
            recon_y = clean_set  # backward-compat alias for the clean-label term

        # KL(ẑ): exponential-family surrogate; per-element log-probs, mean
        # over batch × dim. NOTE : a leading minus made kl_zhat
        # the NEGATIVE KL — e^x - 1 - x is non-negative and equals KL(q||p)
        # in expectation over q — so the optimizer maximized the Student-t
        # divergence and the loss ran away negative. kl_z below is the
        # positive Gaussian KL; consistency requires kl_zhat positive too.
        logp = self._studentt_log_prob(z_hat - z_dec_mu, nu=self.nu0)
        logq = self._studentt_log_prob(z_hat, mu_theta, std_theta,
                                       nu=nu_theta)
        log_p_over_q = logp - logq
        kl_zhat = torch.mean(log_p_over_q.exp() - 1.0 - log_p_over_q)

        # KL(z): standard Gaussian KL vs N(0, I).
        kl_z = -0.5 * torch.mean(
            1.0 + logvar_kappa - mu_kappa.square() - logvar_kappa.exp())

        # Beta IS the KL coefficient (paper eq:correction_loss): recon is
        # unweighted, clean_set carries λ, and β scales the two KL terms.
        loss = recon_loss + clean_set + beta * (kl_zhat + kl_z)
        return {
            'loss': loss,
            'recon': recon_loss,
            'recon_label': label_loss,
            'recon_yhat': label_loss,
            'clean_set_loss': clean_set,
            'recon_y': recon_y,
            'kl_zhat': kl_zhat,
            'kl_z': kl_z,
        }

    # ── IW-ELBO log-weights (retained for eval / inference) ───────────

    def iw_log_weights(
        self, x: torch.Tensor, z_vae: torch.Tensor, y_hat: torch.Tensor,
        K: int, y_clean: torch.Tensor | None = None,
        beta: float = 1.0,
        x_pool: "torch.Tensor | ImagePass | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K importance log-weights per example (Eq. C.5 / C.9).

        ``beta`` is the fractional posterior exponent (paper's $\beta$).
        When ``beta < 1.0``, the likelihood terms are tempered, flattening
        the data influence relative to the prior.  The joint is
            p_β = p(ŷ|x,ẑ)^β · p(x|z)^β · p(ẑ|z) · p(z)        (unsupervised)
            p_β = p(y|x,z)^β · p(ŷ|x,ẑ)^β · p(x|z)^β · p(ẑ|z) · p(z)
                                                              (semi-supervised)
        matching the paper's eq:tempered_joint_app / eq:tempered_joint_semi:
        likelihood terms are tempered, prior and shift terms are not.

        Returns ``(log_w, z_samples)`` where ``log_w`` is (B, K) and
        ``z_samples`` is (K, B, latent_dim).
        """
        if K < 1:
            raise ValueError("K must be at least 1.")
        # Encoder input: the 4D image (image data) or the flat feature vector.
        x_enc = _encoder_input(x)
        img_pass = self.image_pass(x_enc, x_pool)
        # Posterior params (deterministic given the batch).
        mu_theta, logvar_theta, nu_theta = self.noisy_encoder(
            x_enc, self._posterior_label(y_hat), img_pass=img_pass)  # q(ẑ|x,ŷ)
        scale_theta = self._logvar_to_scale(logvar_theta)

        # Hoist label_head.embed_x once: the input encoder is the heavy
        # op on image backbones and is invariant across the K samples.
        x_feat = self.label_head.embed_x(x_enc, img_pass=img_pass)

        cond = self.correction_cond_vector(x_feat, y_hat)

        # Precompute scalar tempering factors (loop-invariant; Python float
        # multiplies fuse into the next elementwise op).
        b = float(beta)
        log_w_list = []
        z_list = []
        for _ in range(K):
            # 1) ẑ ~ q(ẑ|x,ŷ)  (Student-t, rsample)
            z_hat = self._sample_studentt(mu_theta, logvar_theta, nu_theta)
            log_q_zhat = _studentt_logprob(z_hat, mu_theta, scale_theta, nu_theta)

            # 2) z ~ q(z|ẑ, ·)  (Gaussian; gated, blended toward x or toward ŷ)
            mu_kappa, logvar_kappa = self._correction_map(
                z_hat, x_feat, z_vae, y_hat, cond)
            scale_kappa = self._logvar_to_scale(logvar_kappa)
            z = self._sample_normal(mu_kappa, logvar_kappa)
            log_q_z = _normal_logprob(z, mu_kappa, scale_kappa)

            # 3) Generative factors
            logits_hat = self.label_head.head(x_feat, z_hat)
            log_p_yhat = -F.cross_entropy(logits_hat, y_hat, reduction="none")
                
            # log p(x|z) is abandoned; the weights carry the label
            # evidence and the two divergences only.
            mu_psi, logvar_psi = self.shift_model(z)
            scale_psi = self._logvar_to_scale(logvar_psi)
            log_p_zhat_given_z = _studentt_logprob(z_hat, mu_psi, scale_psi, self.nu0)
            log_p_z = _std_normal_logprob(z)

            log_w = (
                b * log_p_yhat
                + log_p_zhat_given_z + log_p_z
                - log_q_zhat - log_q_z
            )

            if y_clean is not None:
                logits_clean = self.label_head.head(x_feat, z)
                log_p_y = -F.cross_entropy(logits_clean, y_clean, reduction="none")
                # Clean-label likelihood is tempered like the other likelihoods
                # (paper eq:tempered_joint_semi raises p(y|x,z) to β).
                log_w = log_w + beta * log_p_y

            log_w_list.append(log_w)
            z_list.append(z)

        log_w = torch.stack(log_w_list, dim=-1)     # (B, K)
        z_samples = torch.stack(z_list, dim=0)       # (K, B, D)
        return log_w, z_samples

    # ── Inference: importance-weighted posterior mean of z ────────────

    @torch.no_grad()
    def sample_corrected_latent(
        self, x: torch.Tensor, z_vae: torch.Tensor, y_hat: torch.Tensor,
        M: int = 5,
        x_pool: "torch.Tensor | ImagePass | None" = None,
    ) -> torch.Tensor:
        """Importance-weighted estimate of z_0 (paper §4 Inference, step 1)."""
        log_w, z_samples = self.iw_log_weights(x, z_vae, y_hat, M,
                                              y_clean=None, x_pool=x_pool)
        return self.importance_weighted_mean(log_w, z_samples)

    @torch.no_grad()
    def corrected_conditioning(
        self, x: torch.Tensor, z0: torch.Tensor,
        z_vae: torch.Tensor | None = None,
        x_pool: "torch.Tensor | ImagePass | None" = None,
    ) -> torch.Tensor:
        """h̃(x, ŷ) — the corrected conditioning signal.

        Returns the paper's ``g_φ(x, z_0)`` — a K-dim class posterior (softmax of the head logits)
        distribution.
        """
        logits = self.corrected_logits(x, z0, x_pool=x_pool)
        return torch.softmax(logits, dim=-1)


def build_lsnpc(
    config,
    *,
    x_dim: int,
    n_classes: int,
    device: str | None = None,
) -> LSNPC:
    """Construct an ``LSNPC`` from a config object (single source of truth).

    Centralises the model-construction kwargs that were previously duplicated
    between ``LSNPCTrainer.train`` (trainers/lsnpc.py) and the checkpoint
    reload path (experiments/lsnpc_stage1.py).  All keyword semantics match
    the former inline constructions exactly; ``device`` optionally places
    the model (``None`` keeps the default placement, mirroring the trainer
    path where ``accel.prepare`` performs the placement).
    """
    model = LSNPC(
        x_dim=int(x_dim),
        latent_dim=int(config.latent_dim),
        n_classes=int(n_classes),
        hidden_dim=int(config.hidden_dim),
        n_blocks=max(1, int(config.n_blocks) - 1),
        image_embed_dim=int(config.lsnpc_embed_dim),
        nu0=float(config.lsnpc_nu0),
        beta=float(config.lsnpc_beta),
        focal_gamma=float(config.lsnpc_focal_gamma),
        focal_alpha=None if config.lsnpc_focal_alpha is None else float(config.lsnpc_focal_alpha),
        image_data=bool(config.lsnpc_image_data),
        img_channels=int(config.lsnpc_img_channels),
        img_size=int(config.lsnpc_img_size),
        encoder_backbone=str(config.encoder_backbone),
        freeze_backbone=bool(config.freeze_backbone),
        head_type=str(config.lsnpc_head),
        correction_cond=str(getattr(config, "lsnpc_correction_cond", "none")),
        shared_yhat_embed=bool(getattr(config, "lsnpc_shared_yhat_embed", False)),
        correction_input=str(getattr(config, "lsnpc_correction_input", "gated")),
    )
    if device is not None:
        model = model.to(device)
    return model
