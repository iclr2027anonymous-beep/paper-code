"""Focal loss on the LSNPC label terms (``lsnpc_focal_gamma`` / ``..._alpha``).

The loss is the one-vs-all form: torchvision's ``sigmoid_focal_loss`` over a
one-hot target, summed over the class axis, so a class index is still the
caller's input. Success criteria, per the codebase rule that a loss switch must
not disturb anything it is not asked to change:

  1. ``gamma == 0`` dispatches to ``F.cross_entropy`` whatever ``alpha``
     says, so every cached cell stays a comparable cross-entropy baseline.
  2. ``gamma > 0`` matches the one-vs-all definition term by term, with
     ``alpha=None`` leaving the terms unweighted.
  3. ``alpha`` weights the terms as ``alpha*y + (1-alpha)*(1-y)``; ``None``
     leaves them unweighted, and ``alpha=1`` is the degenerate end.
  4. Gradients flow, and the focusing term attenuates them near p_t -> 1.
  5. The flags reach the model, and the config rejects out-of-range ``alpha``.
"""
import pytest
import torch
import torch.nn.functional as F

from experiments.arguments import ExperimentConfig
from models.lsnpc import LSNPC, focal_cross_entropy


def _batch(n: int = 64, c: int = 7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g), torch.randint(0, c, (n,), generator=g)


def test_gamma_zero_is_bit_identical_to_cross_entropy():
    logits, target = _batch()
    assert torch.equal(focal_cross_entropy(logits, target, 0.0),
                       F.cross_entropy(logits, target))
    assert torch.equal(
        focal_cross_entropy(logits, target, 0.0, reduction="none"),
        F.cross_entropy(logits, target, reduction="none"))


def _one_vs_all(logits, target, gamma, alpha=None):
    """The definition the loss is contracted to, written out independently."""
    one_hot = F.one_hot(target, logits.size(-1)).to(logits.dtype)
    prob = logits.sigmoid()
    term = F.binary_cross_entropy_with_logits(
        logits, one_hot, reduction="none") * (1.0 - prob * one_hot
                                             - (1.0 - prob) * (1.0 - one_hot)) ** gamma
    if alpha is not None:
        term = term * (alpha * one_hot + (1.0 - alpha) * (1.0 - one_hot))
    return term.sum(dim=-1)


def test_matches_the_one_vs_all_definition():
    logits, target = _batch()
    for gamma in (0.5, 1.0, 2.0):
        assert torch.allclose(
            focal_cross_entropy(logits, target, gamma, alpha=None),
            _one_vs_all(logits, target, gamma).mean())


def test_reduction_none_is_one_loss_per_row():
    logits, target = _batch()
    per_row = focal_cross_entropy(logits, target, 2.0, reduction="none")
    assert per_row.shape == (logits.size(0),)
    assert torch.allclose(per_row.mean(), focal_cross_entropy(logits, target, 2.0))


def test_alpha_weights_target_and_non_target_terms():
    logits, target = _batch()
    for alpha in (0.0, 0.25, 0.5, 0.75):
        assert torch.allclose(
            focal_cross_entropy(logits, target, 2.0, alpha, reduction="none"),
            _one_vs_all(logits, target, 2.0, alpha))


def test_alpha_one_is_the_degenerate_end():
    # alpha = 1 gives every non-target class a zero weight, so only the
    # target-class term survives -- documented, not a usable setting.
    logits, target = _batch()
    one_hot = F.one_hot(target, logits.size(-1)).to(logits.dtype)
    prob = logits.sigmoid()
    target_term = (F.binary_cross_entropy_with_logits(
        logits, one_hot, reduction="none")
        * (1.0 - prob * one_hot - (1.0 - prob) * (1.0 - one_hot)) ** 2.0
        * one_hot).sum(dim=-1)
    assert torch.allclose(
        focal_cross_entropy(logits, target, 2.0, 1.0, reduction="none"),
        target_term)


def test_gamma_zero_dispatches_to_cross_entropy_whatever_alpha_says():
    # gamma=0 means "focal off", and the alpha weighting belongs to the focal
    # form: the switch must not silently rescale the baseline loss of every
    # cached cell because a default alpha happens to be set.
    logits, target = _batch()
    for alpha in (None, 0.0, 0.25, 0.5, 1.0):
        assert torch.equal(focal_cross_entropy(logits, target, 0.0, alpha),
                           F.cross_entropy(logits, target))
        assert torch.equal(
            focal_cross_entropy(logits, target, 0.0, alpha, reduction="none"),
            F.cross_entropy(logits, target, reduction="none"))


def test_focusing_shrinks_a_class_term_relative_to_its_binary_ce():
    # The per-class multiplier is (1 - p_c)^gamma, so a class the model is
    # confident about keeps a smaller fraction of its binary CE.
    logits = torch.tensor([[8.0, -8.0], [0.3, 0.1]])
    target = torch.tensor([0, 1])
    one_hot = F.one_hot(target, 2).float()
    bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    term = _one_vs_all(logits, target, 2.0)
    assert torch.all(term <= bce.sum(dim=-1) + 1e-6)
    ratio = term / bce.sum(dim=-1)
    assert ratio[0] < ratio[1]


def test_gradient_flows_and_focusing_attenuates_it():
    focal_in = torch.tensor([[8.0, -8.0]], requires_grad=True)
    focal_cross_entropy(focal_in, torch.tensor([0]), 2.0).backward()
    ce_in = torch.tensor([[8.0, -8.0]], requires_grad=True)
    F.cross_entropy(ce_in, torch.tensor([0])).backward()
    focal_grad = focal_in.grad.abs().max().item()
    ce_grad = ce_in.grad.abs().max().item()
    assert focal_grad > 0.0
    assert focal_grad < ce_grad


def test_saturated_logits_do_not_produce_nan():
    logits = torch.tensor([[60.0, -60.0]], requires_grad=True)
    out = focal_cross_entropy(logits, torch.tensor([0]), 2.0)
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(logits.grad).all()


def test_unknown_reduction_is_rejected():
    logits, target = _batch()
    with pytest.raises(ValueError, match="reduction"):
        focal_cross_entropy(logits, target, 2.0, reduction="sum")


def test_model_records_the_flags_and_defaults_off():
    kwargs = dict(x_dim=8, latent_dim=4, n_classes=3, hidden_dim=8, n_blocks=2)
    assert LSNPC(**kwargs).focal_gamma == 0.0
    assert LSNPC(**kwargs).focal_alpha is None
    assert LSNPC(**kwargs, focal_gamma=2.0).focal_gamma == 2.0
    assert LSNPC(**kwargs, focal_alpha=0.25).focal_alpha == 0.25


def test_config_rejects_an_out_of_range_alpha():
    ExperimentConfig(lsnpc_focal_alpha=None).validate()
    ExperimentConfig(lsnpc_focal_alpha=0.25).validate()
    with pytest.raises(ValueError, match="lsnpc_focal_alpha"):
        ExperimentConfig(lsnpc_focal_alpha=1.5).validate()
    with pytest.raises(ValueError, match="lsnpc_focal_alpha"):
        ExperimentConfig(lsnpc_focal_alpha=-0.1).validate()


def test_bundle_round_trip_keeps_the_focal_settings(tmp_path):
    from scripts.lsnpc_ckpt import load_bundle, save_bundle

    model = LSNPC(x_dim=8, latent_dim=4, n_classes=3, hidden_dim=8, n_blocks=1,
                  focal_gamma=2.0, focal_alpha=0.25)
    arch = {"build_kwargs": dict(
        x_dim=8, n_classes=3, latent_dim=4, hidden_dim=8, n_blocks=1,
        student_t_nu=5.0, nu0=2.0, beta=1.0, image_data=False, img_channels=3,
        img_size=96, encoder_backbone="conv", freeze_backbone=True,
        focal_gamma=2.0, focal_alpha=0.25)}
    path = save_bundle(tmp_path / "bundle.pt", model, arch=arch,
                       splits={"a": None}, run={}, val_err_h=0.4)
    loaded = load_bundle(path)["model"]
    assert (loaded.focal_gamma, loaded.focal_alpha) == (2.0, 0.25)


def test_legacy_bundles_reconstruct_without_a_focal_alpha():
    from scripts.lsnpc_ckpt import make_config_shim

    kwargs = dict(
        latent_dim=4, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=2.0,
        beta=1.0, image_data=False, img_channels=3, img_size=96,
        encoder_backbone="conv", freeze_backbone=True)
    shim = make_config_shim(kwargs)
    assert (shim.lsnpc_focal_gamma, shim.lsnpc_focal_alpha) == (0.0, None)
    assert make_config_shim(dict(kwargs, focal_alpha=0.25)).lsnpc_focal_alpha == 0.25
