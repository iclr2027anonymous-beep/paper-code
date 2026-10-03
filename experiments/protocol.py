"""Shared CF-protocol computations for the LSNPC label-correction pipelines.

This module is the single home for the must-do protocol computations so every
pipeline (text, image, tabular) computes them identically:

  * AUROC over the misclassified slice (mis-set AUROC)
  * multiclass quality metrics (F1 / precision / recall, per-class mean)
  * symmetric and instance-dependent label-noise injection
  * the four protocol score axes on the eval slice:
      - confidence   (p_max)
      - robustness   (P3': perturb the VAE latent, re-run correction)
      - proximity    (M1 minimality: smallest edit fraction reaching clean)
      - plausibility (log-density of the corrected code under the LSNPC
                      Student-t prior)
    plus per-axis AUROC / valid-mean / invalid-mean reporting.

Pipelines import from here instead of re-implementing; the score KEY NAMES
below are the schema every result.json must use.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from experiments.data import _batched_lsnpc_infer

# ── Score keys (the result.json schema) ────────────────────────────────

CONF_KEY = "p_max(conf)"
# Reported as a robustness control rather than a scored property (paper
# app:score_inventory): its mis-set AUROC is a monotone complement of
# confidence's, so the threshold rule runs on the four scored properties.
ENTROPY_KEY = "entropy"
MINIMALITY_KEY = "minimality(1-minf)"
ROBUSTNESS_KEY = "noise_robustness"
PLAUSIBILITY_KEY = "plausibility(density)"
# Plausibility is scored under the encoder posterior q(ẑ|x,ŷ) = T_ν(μ_θ, σ_θ)
# evaluated at the corrected code (paper eq:plaus_score). The former Student-t
# prior variant and its ``PLAUS_POSTERIOR`` env switch were removed: the prior
# mode is not what the paper reports, and leaving it as the *default* silently
# emitted a different plausibility column whenever the variable was unset.
SCORE_KEYS = (CONF_KEY, ENTROPY_KEY, MINIMALITY_KEY, ROBUSTNESS_KEY,
              PLAUSIBILITY_KEY)

# ── AUROC over the misclassified slice ────────────────────────────────


def mis_set_auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """AUROC of a per-query score against a binary valid indicator.

    Delegates to ``sklearn.metrics.roc_auc_score`` (tie-corrected,
    Wilcoxon-style). Returns NaN when either class is absent.
    """
    s = np.asarray(score, dtype=np.float64)
    p = np.asarray(pos, dtype=bool)
    if p.sum() == 0 or (~p).sum() == 0:
        return float("nan")
    return roc_auc_score(p, s)


def mis_set_ap(score: np.ndarray, pos: np.ndarray) -> float:
    """Average precision of a per-query score against a binary valid
    indicator — precision at high-score thresholds, the decision-relevant
    complement to the AUROC (what a selective admission rule actually
    delivers). Higher score must mean more valid. NaN when either class
    is absent.
    """
    s = np.asarray(score, dtype=np.float64)
    p = np.asarray(pos, dtype=bool)
    if p.sum() == 0 or (~p).sum() == 0:
        return float("nan")
    return average_precision_score(p, s)


# ── Multiclass quality metrics ─────────────────────────────────────────


def multiclass_f1(pred: np.ndarray, y: np.ndarray) -> float:
    """Multiclass F1 (per-class mean), zero-division-safe."""
    if len(pred) == 0:
        return float("nan")
    _, _, f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0)
    return f1  # sklearn already returns a plain Python float


def multiclass_precision(pred: np.ndarray, y: np.ndarray) -> float:
    """Multiclass precision (per-class mean), zero-division-safe."""
    if len(pred) == 0:
        return float("nan")
    p, _, _, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0)
    return p


def multiclass_recall(pred: np.ndarray, y: np.ndarray) -> float:
    """Multiclass recall (per-class mean), zero-division-safe."""
    if len(pred) == 0:
        return float("nan")
    _, r, _, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0)
    return r


# ── Label-noise injection ──────────────────────────────────────────────


def inject_noise(y, rate: float, seed: int, n_classes: int):
    """Symmetric multiclass label noise: flip each label to a uniformly
    random OTHER class with probability ``rate``.

    Effective flip fraction equals ``rate`` (no self-flips): a flipped
    row's new class is drawn uniformly from the ``n_classes - 1`` classes
    different from its original label.
    """
    y = np.array(y).copy()
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y)) < rate
    idx = np.where(flip)[0]
    if len(idx) == 0:
        return y
    # Uniform shift in {1, ..., n_classes-1}: maps every flipped row to a
    # different class (no self-flips by construction).
    shift = rng.integers(1, n_classes, size=len(idx))
    y[idx] = (y[idx] + shift) % n_classes
    return y


# Row-chunk size for the (n, n_classes, D) distance intermediate in
# ``inject_noise_idn``. ``0`` (default) reproduces the historical
# single-shot computation -- the whole (n, n_classes, D) array at once.
# A positive row count caps that intermediate at (chunk, n_classes, D),
# which is what keeps 64x64x3 EuroSAT features (~10.6 GB at n = n_classes =
# full pool) inside memory. Only the row slicing changes: the distance
# expression itself is untouched, so every entry stays bit-identical to
# the single-shot result. Read only by ``inject_noise_idn``.
IDN_DIST_CHUNK = 0


def _centroid_dists(X: np.ndarray, centroids: np.ndarray,
                    chunk_rows: int = 0) -> np.ndarray:
    """(n, n_classes) distances from every row of ``X`` to every centroid.

    The same expression as ``np.linalg.norm(X[:, None, :]
    - centroids[None, :, :], axis=2)`` applied to row chunks of at most
    ``chunk_rows`` rows (``<= 0``: all rows in one shot). The subtraction
    and the reduction are per-row elementwise/reduce ops, so a row slice
    cannot change a single floating value -- it only bounds the size of
    the (chunk, n_classes, D) intermediate. The algebraic expansion
    ``||x||^2 - 2 x.c + ||c||^2`` is deliberately NOT used: it is a
    different floating-point path.
    """
    n = len(X)
    step = n if chunk_rows <= 0 else int(chunk_rows)
    if step >= n:
        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.concatenate(
        [np.linalg.norm(
            X[i0:i0 + step, None, :] - centroids[None, :, :], axis=2
         )
         for i0 in range(0, n, step)], 
        axis=0
    )


def inject_noise_idn(y, rate: float, seed: int, n_classes: int, X=None):
    """Instance-dependent multiclass label noise.

    Flip probability per sample scales with feature proximity to the
    other class centroids: samples lying close to a competing class (or
    far from their own) are more likely to be mislabelled into it, while
    clear samples are left alone. Per-sample probabilities are calibrated
    so the overall flip fraction ≈ ``rate``. Flip targets are
    distance-weighted over the other classes (nearer centroids more
    likely), never the true class.

    ``X`` is the feature matrix used to estimate class centroids. Falls
    back to symmetric noise if ``X`` is None.
    """
    y = np.array(y).copy()
    if X is None or rate <= 0:
        return inject_noise(y, rate, seed, n_classes)
    rng = np.random.default_rng(seed)
    n = len(y)
    # Class centroids from the (clean) labels.
    centroids = np.stack([X[y == c].mean(axis=0) for c in range(n_classes)])
    # Distances of every row to every centroid: (n, n_classes). Materialised
    # in row chunks of IDN_DIST_CHUNK rows (0 = all rows at once, the
    # historical behaviour); same expression, same values, smaller peak.
    dists = _centroid_dists(X, centroids, IDN_DIST_CHUNK)
    d_self = dists[np.arange(n), y]
    d_other = dists.copy()
    d_other[np.arange(n), y] = np.inf
    d_min_other = d_other.min(axis=1)
    # Difficulty in [0, 1]: 0 = row sits on its own centroid (never flip);
    # -> 1 as a competing centroid is closer than the true one.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d_self / (d_self + d_min_other)
    # Calibrate so the mean per-row flip probability == rate.
    scale = rate / np.clip(t.mean(), 1e-9, None)
    p = np.clip(t * scale, 0.0, 1.0)
    flip = rng.random(n) < p
    idx = np.where(flip)[0]
    if len(idx) == 0:
        return y
    # Flip target: softmax over distances to the OTHER classes' centroids.
    # Per-row temperature = median of the finite other-class distances
    # (binary case: the single other class).
    with np.errstate(divide="ignore", invalid="ignore"):
        med_other = np.median(np.where(d_other == np.inf, np.nan, d_other), axis=1)
    med_other = np.where(np.isnan(med_other), 1.0, med_other)
    for i in idx:
        c = int(y[i])
        d = dists[i].copy()
        d[c] = np.inf
        w = np.exp(-d / (med_other[i] + 1e-9))
        w[c] = 0.0
        w /= w.sum()
        y[i] = rng.choice(n_classes, p=w)
    return y


# ── Protocol scores on the eval slice ──────────────────────────────────


def _precompute_image_pool(lsnpc, Xt: torch.Tensor,
                           batch_size: int) -> torch.Tensor | None:
    """Pooled frozen-trunk features for every row of ``Xt``, or ``None``.

    The scorers re-encode the SAME eval images on every perturbation draw (K
    per row), every minimality fraction and the plausibility pass, and a
    frozen trunk in eval mode is invariant to all of it. Pooling once and
    handing the slices to the encoder calls removes that repetition without
    changing a number. Sliced by index rather than by DataLoader: this pass
    runs before the seeded draws below and must advance no RNG stream at all.
    Returns ``None`` for a model that does not share one frozen trunk (text
    data, trainable backbone).
    """
    if not getattr(lsnpc, "share_backbone_pass", False):
        return None
    parts = []
    for i in range(0, len(Xt), batch_size):
        xb = Xt[i:i + batch_size]
        parts.append(lsnpc.image_pool(xb))
    return torch.cat(parts, dim=0)


def _perturbation_noise(zv: torch.Tensor, K: int, seed: int) -> torch.Tensor:
    """``(K, n, D)`` latent perturbations: one torch draw on ``zv``'s device.

    The K draws used to come from ``np.random.randn`` one at a time, so the
    robustness column rode the *global* NumPy stream (which every DataLoader
    base seed also advances) and crossed the bus once per draw. An explicit
    generator keeps the flips off both the global streams, and the whole
    ensemble is one tensor the sampler can take in a single sweep.
    """
    gen = torch.Generator(device=zv.device)
    gen.manual_seed(seed)
    scale = 0.1 * float(zv.std())
    return zv.unsqueeze(0) + scale * torch.randn(
        (K, *zv.shape), generator=gen, device=zv.device, dtype=zv.dtype)


@torch.no_grad()
def _perturbed_latent_labels(lsnpc, Xt: torch.Tensor, yh_t: torch.Tensor,
                             zv_pert: torch.Tensor, *, x_pool, M: int,
                             batch_size: int, device) -> torch.Tensor:
    """Correction labels for ``(K, n, D)`` perturbed latents, returned ``(K, n)``.

    The K draws ride in the row batch: one pass over the eval slice carries the
    whole K-perturbation ensemble, instead of K whole-slice passes. Each step
    holds ``batch_size`` rows in total -- K draws of ``batch_size // K`` latent
    rows -- so peak activation memory stays at the caller's batch size rather
    than K times it (the trainable-backbone arm already runs at the ceiling).
    The latent row slice is still the original index range, so ``x_pool``
    slicing (and the pooled-trunk reuse) is unchanged.
    """
    K, n, latent = zv_pert.shape
    rows = max(1, batch_size // K)
    labels = torch.empty((K, n), dtype=torch.long, device=device)
    for i0 in range(0, n, rows):
        sl = slice(i0, min(i0 + rows, n))
        xb = Xt[sl]
        b = xb.shape[0]
        xb = xb.repeat(K, *([1] * (xb.dim() - 1)))
        yb = yh_t[sl].repeat(K)
        zb = zv_pert[:, sl].reshape(K * b, latent)
        pool = None if x_pool is None else x_pool[sl].repeat(K, 1)
        zc = lsnpc.sample_corrected_latent(xb, zb, yb, M=M, x_pool=pool)
        logits = lsnpc.corrected_conditioning(xb, zc, x_pool=pool)
        labels[:, sl] = logits.argmax(dim=-1).reshape(K, b)
    return labels


@torch.no_grad()
def _conditioning_labels(lsnpc, Xt: torch.Tensor, z: torch.Tensor, *,
                         x_pool, batch_size: int, device) -> torch.Tensor:
    """Corrected conditioning labels for a batch of latents, shape ``(n,)``.

    Index-sliced rather than DataLoader-iterated, like
    ``_precompute_image_pool``: the latents already live on the device, so a
    loader would only add a per-batch host copy and an iterator base seed. The
    argmax stays on the device; callers move the labels if they want them in
    NumPy.
    """
    labels = torch.empty(len(z), dtype=torch.long, device=device)
    for i0 in range(0, len(z), batch_size):
        sl = slice(i0, min(i0 + batch_size, len(z)))
        pool = None if x_pool is None else x_pool[sl]
        labels[sl] = lsnpc.corrected_conditioning(
            Xt[sl], z[sl], x_pool=pool).argmax(dim=-1)
    return labels


@torch.no_grad()
def _posterior_plausibility(lsnpc, Xt: torch.Tensor, yh_t: torch.Tensor,
                            zc_t: torch.Tensor, *, x_pool, batch_size: int,
                            device) -> torch.Tensor:
    """``posterior_log_prob`` over the eval slice, shape ``(n,)`` on ``device``.

    Same index slicing as the other passes: with no iterator there is no base
    seed to draw, so the scoring entry points no longer have to route around
    the global RNG to keep this pass off it.
    """
    pl = torch.zeros(len(zc_t), device=device, dtype=torch.float32)
    for i0 in range(0, len(zc_t), batch_size):
        sl = slice(i0, min(i0 + batch_size, len(zc_t)))
        pool = None if x_pool is None else x_pool[sl]
        pl[sl] = lsnpc.posterior_log_prob(Xt[sl], yh_t[sl], zc_t[sl],
                                          x_pool=pool)
    return pl


def _confidence(probs: np.ndarray) -> np.ndarray:
    """``p_max`` of the corrected softmax: the confidence axis."""
    return probs.max(-1)


def _minimality_grid(lsnpc, Xt: torch.Tensor, zv_t: torch.Tensor,
                     z_corr_t: torch.Tensor, ref, *, x_pool, batch_size: int,
                     device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``minimality(1-minf)`` against ``ref``, over the fraction grid.

    The smallest fraction of the latent edit whose re-run corrected
    conditioning still reproduces ``ref``: the row's own corrected label for
    the ``_self`` axis, the clean label for the oracle axis. Returns the score
    and the per-fraction ``valid_at_<pct>`` masks, which the oracle scorer
    reports as its own columns.
    """
    ref_t = torch.as_tensor(ref, device=device)
    shift = z_corr_t - zv_t
    minf = np.full(len(ref), 1.0)
    valid_at = {}
    for f in (0.25, 0.5, 0.75, 1.0):
        ok = (_conditioning_labels(lsnpc, Xt, zv_t + f * shift,
                                   x_pool=x_pool, batch_size=batch_size,
                                   device=device) == ref_t).cpu().numpy()
        valid_at[f"valid_at_{int(f * 100)}"] = ok
        minf[ok] = np.minimum(minf[ok], f)
    return 1.0 - minf, valid_at


def _robustness_labels(lsnpc, Xt: torch.Tensor, yh_t: torch.Tensor,
                       zv_t: torch.Tensor, K: int, seed: int, *, x_pool,
                       M: int, batch_size: int, device) -> torch.Tensor:
    """``(K, n)`` corrected labels for K seeded perturbations of the VAE latent.

    The sampler stream is reset to ``seed`` first, so the ensemble does not
    inherit how many draws the inference above consumed; the flips come from
    their own generator and are unaffected by either.
    """
    torch.manual_seed(seed)
    return _perturbed_latent_labels(
        lsnpc, Xt, yh_t, _perturbation_noise(zv_t, K, seed), x_pool=x_pool,
        M=M, batch_size=batch_size, device=device)


def _preservation_rate(labels: torch.Tensor, ref, K: int) -> np.ndarray:
    """Fraction of the K perturbation draws whose correction still returns ``ref``."""
    hits = (labels == torch.as_tensor(ref, device=labels.device)).sum(dim=0)
    return (hits.double() / K).cpu().numpy()


@torch.no_grad()
def compute_protocol_scores(
    lsnpc, X_eval: np.ndarray, y_clean: np.ndarray, y_noisy: np.ndarray,
    device: str, batch_size: int = 256, M: int = 5,
    seed: int = 0, which: tuple[str, ...] = SCORE_KEYS,
    z_vae: np.ndarray | None = None,
    k_perturb: int = 4,
) -> dict:
    """Compute the four protocol score axes on the eval slice.

    Args:
        lsnpc: trained LSNPC model (eval mode).
        X_eval: (n, ...) eval features. For image data pass the model's own
            x tensor layout (NHWC float); ``z_vae`` then supplies the VAE
            latents separately. For embedding/tabular data X_eval is the
            flat feature matrix and z_vae may stay None (z = x).
        y_clean: (n,) clean labels (the oracle).
        y_noisy: (n,) noisy labels.
        device: torch device string.
        batch_size: inference batch size.
        M: importance-weight samples for the correction path.
        seed: RNG seed for the robustness perturbation (P3').
        z_vae: optional (n, latent) VAE latents. Required when the LSNPC x
            path and the VAE latent differ (image pipelines where x is the
            pixel tensor); when None, z_vae = X_eval flattened (the text
            embedding contract).
        k_perturb: number of perturbation draws for the noise-robustness
            score (K). Legacy runs used K=4; the protocol (Set G settings)
            uses K=32. A larger K tightens the per-query estimate; the K
            draws share one pass over the eval slice at the caller's batch
            size, so K does not multiply the sweep count.

    Returns a dict with:
        'corr': corrected labels (n,)
        'probs': corrected softmax (n, n_classes)
        'scores': {SCORE_KEY -> (n,) per-query score array} (incl. valid_at_*)
        'rows': list of (name, mis-set AUROC, valid-mean, invalid-mean)
        'mis': bool mask of noisy != clean
        'succ': bool mask of corr == clean
        'z_corr': corrected codes (n, latent_dim)
    """
    Xt = torch.as_tensor(X_eval, dtype=torch.float32, device=device)
    if z_vae is not None:
        zv_t = torch.as_tensor(z_vae, dtype=torch.float32, device=device)
    else:
        zv_t = torch.as_tensor(
            np.asarray(X_eval).reshape(len(X_eval), -1),
            dtype=torch.float32, device=device)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long,
                           device=device)
    X_pool = _precompute_image_pool(lsnpc, Xt, batch_size)
    torch.manual_seed(seed)
    z_corr_t, c_corr_t = _batched_lsnpc_infer(
        lsnpc, Xt, zv_t, yh_t, batch_size, device, M, x_pool=X_pool,
    )
    z_corr = z_corr_t.cpu().numpy()
    c_corr = c_corr_t.cpu().numpy()

    corr = np.argmax(c_corr, axis=-1)
    probs = c_corr  # corrected_conditioning already returns softmax
    pmax = _confidence(probs)
    # Softmax entropy of the corrected class distribution (lower is better).
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)

    # M1 minimality + valid_at fractions (proximity axis): smallest fraction
    # of the LATENT edit that still reaches the clean label. The edit is
    # applied in z_vae, not x: for text/embedding pipelines z == x, but for
    # images the two live in different spaces.
    minimality = None
    valid_at_f = {}
    if MINIMALITY_KEY in which:
        minimality, valid_at_f = _minimality_grid(
            lsnpc, Xt, zv_t, z_corr_t, y_clean, x_pool=X_pool,
            batch_size=batch_size, device=device)

    # P3' noise robustness: perturb the VAE latent, re-run correction.
    nr = None
    if ROBUSTNESS_KEY in which:
        K = int(k_perturb)
        labels = _robustness_labels(
            lsnpc, Xt, yh_t, zv_t, K, seed, x_pool=X_pool, M=M,
            batch_size=batch_size, device=device)
        nr = _preservation_rate(labels, y_clean, K)

    # Plausibility: log-density of the corrected code under the encoder
    # posterior q(ẑ|x,ŷ) = T_nu(mu_theta, sigma_theta) (paper eq:plaus_score).
    pl = None
    if PLAUSIBILITY_KEY in which:
        zc_t = torch.as_tensor(z_corr, dtype=torch.float32, device=device)
        pl = _posterior_plausibility(
            lsnpc, Xt, yh_t, zc_t, x_pool=X_pool, batch_size=batch_size,
            device=device).cpu().numpy()

    mis = y_noisy != y_clean
    succ = corr == y_clean

    # ── Transition categories (the 2x2 of noisy-correct x corrected-correct):
    #   WR = wrong->right  (repair: the correction fixed a noisy-wrong label)
    #   RW = right->wrong  (damage: the correction broke a noisy-correct label)
    #   RR = right->right  (kept)
    #   WW = wrong->wrong  (still wrong)
    # The existing mis-set AUROC measures the repair channel only (among
    # `mis`); the damage channel (among ~mis, was-right queries) needs its
    # own per-axis AUROC: does a high score predict the correction BREAKING
    # a correct label? Net value of applying the correction =
    # P(repair) - P(damage), which is what an admission gate must weigh.
    trans = {
        "WR": int((mis & succ).sum()),
        "RW": int((~mis & ~succ).sum()),
        "RR": int((~mis & succ).sum()),
        "WW": int((mis & ~succ).sum()),
    }
    was_right = ~mis
    broke = ~succ & was_right  # right->wrong

    scores = {}
    if CONF_KEY in which:
        scores[CONF_KEY] = pmax
    if ENTROPY_KEY in which:
        scores[ENTROPY_KEY] = entropy
    if MINIMALITY_KEY in which:
        scores[MINIMALITY_KEY] = minimality
    if ROBUSTNESS_KEY in which:
        scores[ROBUSTNESS_KEY] = nr
    if PLAUSIBILITY_KEY in which:
        scores[PLAUSIBILITY_KEY] = pl
    scores.update(valid_at_f)

    rows = []
    for name, s in scores.items():
        # AP is direction-sensitive: entropy is lower-is-better, everything
        # else higher-is-better. Negate entropy so higher score = more valid.
        # The repair AUROC stays on the raw score, which is why the reported
        # entropy AUROC reads as the complement of confidence's.
        s_for_ap = -s if name == ENTROPY_KEY else s
        # Repair channel: among noisy-wrong, does the score rank repaired
        # queries higher? (existing mis-set AUROC)
        repair_auroc = mis_set_auroc(s[mis].astype(float), succ[mis])
        repair_ap = mis_set_ap(s_for_ap[mis].astype(float), succ[mis])
        # Damage channel: among noisy-correct, does a HIGH score predict the
        # correction BREAKING the label? A risk AUROC > 0.5 means the score
        # flags fragile corrections — admit-inverted for gating.
        damage_auroc = mis_set_auroc(s[was_right].astype(float), broke[was_right])
        damage_ap = mis_set_ap(s[was_right].astype(float), broke[was_right])
        rows.append((
            name,
            repair_auroc,
            repair_ap,
            np.mean(s[succ & mis]) if succ[mis].any() else float("nan"),
            np.mean(s[(~succ) & mis]) if (~succ & mis).any() else float("nan"),
            damage_auroc,
            damage_ap,
        ))
    rows.sort(key=lambda r: -abs(r[1] - 0.5))

    return {
        "corr": corr,
        "probs": probs,
        "scores": scores,
        "rows": rows,
        "mis": mis,
        "succ": succ,
        "z_corr": z_corr,
        "transitions": trans,
        "repair_auroc": {r[0]: r[1] for r in rows},
        "damage_auroc": {r[0]: r[5] for r in rows},
        "damage_ap": {r[0]: r[6] for r in rows},
    }


# ── Oracle-free stream scoring (Set E emissions, ENG-2) ────────────────


@torch.no_grad()
def compute_stream_scores(
    lsnpc,
    X_eval: np.ndarray,
    y_noisy: np.ndarray,
    device: str,
    batch_size: int = 256,
    M: int = 5,
    seed: int = 0,
    which: tuple[str, ...] = SCORE_KEYS,
    z_vae: np.ndarray | None = None,
    k_perturb: int = 32,
    y_clean: np.ndarray | None = None,
) -> dict:
    """Per-row correction scores on an ARBITRARY pool (Set E emission mode).

    This is the oracle-free counterpart of ``compute_protocol_scores``: every
    score is a function of the correction path alone (methodology §4 — scores
    are "computed from the correction path, never from a clean label"):

      - ``p_max(conf)``: max-softmax of the corrected conditioning
        g_phi(x, z_0);
      - ``minimality(1-minf)_self``: smallest grid fraction of the latent
        edit whose re-run corrected conditioning reproduces the row's own
        corrected label ĉ (eq:minf with the clean label replaced by ĉ — the
        counterfactual target of the edit itself; usable on unlabelled rows);
      - ``noise_robustness_self``: K-perturbation label-preservation rate
        against ĉ (eq:rob_score — re-decodes to ŷ(x));
      - ``plausibility(density)``: Student-t log-density of the corrected
        code (eq:plaus_score).

    When ``y_clean`` IS supplied (rows carry a clean oracle), the clean-
    referenced protocol scores (the evaluation instruments of
    ``compute_protocol_scores``) are also returned under the
    ``minimality(1-minf)`` / ``noise_robustness`` keys plus ``mis`` /
    ``succ`` / ``transitions`` — stored for DIAGNOSTICS and the held-out
    label-level sign check only, never used by the Set E gate (which ranks by
    the ``_self`` scores).

    Returns a dict with ``corr`` (corrected labels), ``probs``,
    ``edited`` (corr != y_noisy), and ``scores`` (one (n,) array per axis;
    keys use the SCORE_KEYS names where the definition is unchanged and
    ``_self``-suffixed keys for the self-consistent gate axes).
    """
    n = len(X_eval)
    Xt = torch.as_tensor(X_eval, dtype=torch.float32, device=device)
    if z_vae is not None:
        zv_t = torch.as_tensor(z_vae, dtype=torch.float32, device=device)
    else:
        zv_t = torch.as_tensor(np.asarray(X_eval).reshape(n, -1),
                               dtype=torch.float32, device=device)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long,
                           device=device)
    X_pool = _precompute_image_pool(lsnpc, Xt, batch_size)
    torch.manual_seed(seed)
    z_corr_t, c_corr_t = _batched_lsnpc_infer(
        lsnpc, Xt, zv_t, yh_t, batch_size, device, M, x_pool=X_pool,
    )
    z_corr = z_corr_t.cpu().numpy()
    c_corr = c_corr_t.cpu().numpy()
    corr = np.argmax(c_corr, axis=-1)
    pmax = _confidence(c_corr)
    entropy = -np.sum(c_corr * np.log(np.clip(c_corr, 1e-12, 1.0)), axis=-1)

    # Plausibility: log-density of the corrected code under the encoder
    # posterior q(ẑ|x,ŷ) = T_nu(mu_theta, sigma_theta) (paper eq:plaus_score).
    zc_t = torch.as_tensor(z_corr, dtype=torch.float32, device=device)
    pl = _posterior_plausibility(
        lsnpc, Xt, yh_t, zc_t, x_pool=X_pool, batch_size=batch_size,
        device=device).cpu().numpy()

    scores = {CONF_KEY: np.asarray(pmax, dtype=np.float64),
              ENTROPY_KEY: np.asarray(entropy, dtype=np.float64),
              PLAUSIBILITY_KEY: np.asarray(pl, dtype=np.float64)}

    grid_kw = dict(x_pool=X_pool, batch_size=batch_size, device=device)

    if MINIMALITY_KEY in which:
        self_min, _ = _minimality_grid(lsnpc, Xt, zv_t, z_corr_t, corr,
                                       **grid_kw)
        scores["minimality(1-minf)_self"] = self_min
        if y_clean is not None:
            oracle_min, _ = _minimality_grid(
                lsnpc, Xt, zv_t, z_corr_t, np.asarray(y_clean), **grid_kw)
            scores[MINIMALITY_KEY] = oracle_min

    if ROBUSTNESS_KEY in which:
        K = int(k_perturb)
        # See ``compute_protocol_scores``: the sampler stream is reset inside
        # ``_robustness_labels``, the flips are one torch draw on the model's
        # device, and all K of them go through the correction in one sweep.
        labels = _robustness_labels(lsnpc, Xt, yh_t, zv_t, K, seed,
                                    M=M, **grid_kw)
        scores["noise_robustness_self"] = _preservation_rate(labels, corr, K)
        if y_clean is not None:
            scores[ROBUSTNESS_KEY] = _preservation_rate(
                labels, np.asarray(y_clean), K)

    out = {"corr": corr, "probs": c_corr, "edited": corr != np.asarray(y_noisy),
           "z_corr": z_corr, "scores": scores}
    if y_clean is not None:
        yc = np.asarray(y_clean)
        mis = y_noisy != yc
        succ = corr == yc
        out.update(mis=mis, succ=succ)
        out["transitions"] = {
            "WR": int((mis & succ).sum()), "RW": int((~mis & ~succ).sum()),
            "RR": int((~mis & succ).sum()), "WW": int((mis & ~succ).sum())}
    return out
