"""The gated two-pathway label head and the posterior's label split.

  * ``head_type="gate"`` replaces the concat head with two per-input trunks
    mixed by a learned gate, ``lambda * f_x(x) + (1 - lambda) * f_z(z)``, and
    records ``last_gate`` so reliance on the image pathway is observable.
  * ``correction_cond`` selects what the correction map reads: the VAE latent
    of x (``none``), the image embedding (``x``), both (``x_yhat``), or a
    learned label embedding in place of x (``yhat``).

All default to the prior behaviour, so cached artifacts reproduce exactly; the
data-reconstruction term is gone (removed after S3l) and must not reappear.
"""
from types import SimpleNamespace

import pytest
import torch

from models.lsnpc import LSNPC, build_lsnpc
from scripts.lsnpc_ckpt import make_config_shim


def _model(head_type="concat", image_data=False, dropout=0.0,
           x_dim=4, correction_cond="none"):
    torch.manual_seed(0)
    return LSNPC(
        x_dim=x_dim, latent_dim=3, n_classes=2, hidden_dim=8, n_blocks=1,
        dropout=dropout,
        image_data=image_data, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=False,
        head_type=head_type,
        correction_cond=correction_cond,
    )


def _flat_batch(B=8, x_dim=4, latent_dim=3, n_classes=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, x_dim, generator=g),
            torch.randn(B, latent_dim, generator=g),
            torch.randint(0, n_classes, (B,), generator=g))


def _image_batch(B=6, latent_dim=3, n_classes=2, seed=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 3, 32, 32, generator=g),
            torch.randn(B, latent_dim, generator=g),
            torch.randint(0, n_classes, (B,), generator=g))


# ── gated two-pathway head ───────────────────────────────────────────────

def test_gate_head_is_a_convex_mixture_and_records_lambda():
    model = _model(head_type="gate").eval()
    x, z, _ = _flat_batch()
    logits = model.label_head(x, z)
    assert logits.shape == (8, 2)
    lam = model.label_head.last_gate
    assert isinstance(lam, float) and 0.0 < lam < 1.0


def test_gate_head_lambda_one_reproduces_the_image_trunk():
    model = _model(head_type="gate").eval()
    x, z, _ = _flat_batch()
    with torch.no_grad():
        head = model.label_head
        head.gate_x.weight.zero_()
        head.gate_x.bias.fill_(50.0)   # sigmoid(50) ~ 1
        logits = head(x, z)
        x_only = head.x_out(head.x_trunk(head.embed_x(x)))
    assert torch.allclose(logits, x_only, atol=1e-5)
    assert model.label_head.last_gate > 0.99


def test_gate_head_defaults_off_and_gradients_reach_both_paths():
    model = _model(head_type="gate").train()
    x, z, _ = _flat_batch()
    model.label_head(x, z).sum().backward()
    for name, param in model.label_head.named_parameters():
        if name.startswith(("x_trunk", "z_trunk", "x_out", "z_out",
                            "gate_x", "gate_z")):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_concat_head_carries_no_gate_state():
    model = _model(head_type="concat").eval()
    assert not hasattr(model.label_head, "last_gate")
    assert not hasattr(model.label_head, "gate_x")
    x, z, _ = _flat_batch()
    assert model.label_head(x, z).shape == (8, 2)


def test_unknown_head_type_is_rejected():
    with pytest.raises(ValueError, match="concat"):
        _model(head_type="mixture")


def test_gate_head_changes_the_parameter_count_but_not_forward_shape():
    concat = _model(head_type="concat")
    gate = _model(head_type="gate")
    x, z, _ = _flat_batch()
    assert concat.label_head(x, z).shape == gate.label_head(x, z).shape
    assert sum(p.numel() for p in gate.label_head.parameters()) != \
        sum(p.numel() for p in concat.label_head.parameters())


def _loss(model, x, z_vae, y_hat, beta=1.0):
    return model.compute_loss(x, y_hat, beta=beta, z_vae=z_vae, eta=0.0)


def test_the_objective_carries_no_data_reconstruction_term():
    """The removed term must not reappear in the loss dict or the total."""
    model = _model(image_data=True).eval()
    x, z, y = _image_batch()
    out = _loss(model, x, z, y)
    assert "recon_image" not in out
    assert out["recon"] is out["recon_label"]
    assert torch.isfinite(out["loss"])


# ── plumbing: build_lsnpc, config shim, bundle round-trip ────────────────

def _shim(**overrides):
    fields = dict(
        latent_dim=3, hidden_dim=8, n_blocks=1, lsnpc_nu0=5.0,
        lsnpc_beta=1.0, lsnpc_focal_gamma=0.0,
        lsnpc_image_data=False, lsnpc_img_channels=3, lsnpc_img_size=32,
        encoder_backbone="conv", freeze_backbone=True,
        lsnpc_head="concat", lsnpc_embed_dim=128, lsnpc_focal_alpha=None,
        lsnpc_correction_cond="none",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_build_lsnpc_forwards_the_head_knob():
    model = build_lsnpc(_shim(lsnpc_head="gate"), x_dim=4, n_classes=2)
    assert model.label_head.head_type == "gate"


# ── one build schema: config -> bundle -> shim -> model ─────────────────

def test_every_schema_row_names_a_real_config_field():
    """A row naming a field ExperimentConfig lost would silently fall back to
    the row's default and rebuild the wrong model."""
    from experiments.arguments import ExperimentConfig
    from scripts.lsnpc_ckpt import LSNPC_BUNDLE_FIELDS

    cfg = ExperimentConfig()
    for key, attr, _default, _cast in LSNPC_BUNDLE_FIELDS:
        assert hasattr(cfg, attr), f"{key} -> {attr} is not a config field"


def test_the_bundle_schema_round_trips_every_structural_knob():
    """The three schema sites used to be written out separately and drifted
    twice: a shim demanding a ``decoder_type`` key nobody wrote (every bundle
    load raised) and a focal alpha the bundle never recorded (a focal run
    reloaded as plain cross-entropy). This is the round trip that pins them."""
    from experiments.arguments import ExperimentConfig
    from scripts.lsnpc_ckpt import LSNPC_BUNDLE_FIELDS, bundled_build_kwargs

    cfg = ExperimentConfig(lsnpc_correction_cond="x_yhat",
                           lsnpc_shared_yhat_embed=True,
                           lsnpc_correction_input="concat",
                           lsnpc_head="gate", lsnpc_embed_dim=64,
                           lsnpc_focal_gamma=2.0, lsnpc_focal_alpha=0.25)
    kw = bundled_build_kwargs(cfg, x_dim=4, n_classes=2)
    shim = make_config_shim(kw)

    # What the writer recorded is what the shim hands back, key for key.
    for key, attr, _default, _cast in LSNPC_BUNDLE_FIELDS:
        assert getattr(shim, attr) == kw[key], key
    assert shim.n_blocks == cfg.n_blocks
    assert build_lsnpc(shim, x_dim=4, n_classes=2).noisy_encoder.label_dim \
        == build_lsnpc(cfg, x_dim=4, n_classes=2).noisy_encoder.label_dim

    # The model the trainer builds and the model a reload builds must have the
    # same parameter set, or the strict state_dict load cannot succeed.
    from_config = build_lsnpc(cfg, x_dim=4, n_classes=2)
    from_bundle = build_lsnpc(shim, x_dim=4, n_classes=2)
    assert set(from_config.state_dict()) == set(from_bundle.state_dict())
    from_bundle.load_state_dict(from_config.state_dict())


def test_build_lsnpc_projects_the_backbone_feature_to_the_configured_width():
    model = build_lsnpc(_shim(lsnpc_image_data=True, lsnpc_embed_dim=64),
                        x_dim=4, n_classes=2)
    for holder in (model.noisy_encoder, model.label_head):
        # The pretrained backbone's pooled feature is never fed raw: it is
        # projected by the encoder's mu head to the configured width.
        assert holder.image_encoder.mu.out_features == 64
        assert holder.image_encoder.logvar.out_features == 64


def test_default_embed_dim_keeps_the_128_d_projection():
    assert _shim().lsnpc_embed_dim == 128
    model = build_lsnpc(_shim(lsnpc_image_data=True), x_dim=4, n_classes=2)
    assert model.label_head.image_encoder.mu.out_features == 128


def test_config_shim_restores_the_head_knob_and_defaults_old_bundles():
    shim = make_config_shim(dict(
        latent_dim=3, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=5.0,
        beta=1.0, image_data=False, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=True, image_embed_dim=64))
    assert shim.lsnpc_embed_dim == 64
    assert make_config_shim(dict(
        latent_dim=3, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=5.0,
        beta=1.0, image_data=False, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=True)).lsnpc_embed_dim == 128

    # A bundle from the removed data-reconstruction era still loads.
    legacy = make_config_shim(dict(
        latent_dim=3, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=5.0,
        beta=1.0, image_data=False, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=True,
        recon_weight=10.0, head="gate"))
    assert legacy.lsnpc_head == "gate"
    assert not hasattr(legacy, "lsnpc_recon_weight")

    assert make_config_shim(dict(
        latent_dim=3, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=5.0,
        beta=1.0, image_data=False, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=True)).lsnpc_head == "concat"


def test_ckpt_name_records_the_head_token_only_when_enabled():
    from scripts.image_lsnpc import _lsnpc_ckpt_name

    config = SimpleNamespace(lsnpc_img_size=32)
    args = SimpleNamespace(
        dataset="cifar10n", seed=42, epochs=3, beta=0.5, clean_set_size=2000,
        focal_gamma=0.0, focal_alpha=None, lsnpc_head="concat",
        lsnpc_embed_dim=128, correction_cond="none",
        shared_yhat_embed=False, correction_input="gated",
        eta=0.1, backbone="resnet50")
    assert _lsnpc_ckpt_name(config, args, True, True) == \
        "lsnpc_cifar10n_s42_e3_b0.5_cs2000_eta0.1_train_sz32.pt"

    args.lsnpc_head = "gate"
    name = _lsnpc_ckpt_name(config, args, True, True)
    assert "_hgate" in name

    args.lsnpc_embed_dim = 64
    assert "_ed64" in _lsnpc_ckpt_name(config, args, True, True)

    # An alpha run and an unweighted run of the same cell must not share a
    # name: the token is what keeps their bundles and scratch dirs apart.
    args.focal_alpha = 0.25
    assert "_fa0.25" in _lsnpc_ckpt_name(config, args, True, True)


# ── the posterior's split: label in the trunk, never in the mean ─────────

def test_the_mean_is_label_free_and_the_shape_is_not():
    """The var_shared invariant: the label conditions the trunk -- and with it
    both shape parameters, logvar and the df -- and never the location, which
    is the one parameter the decoder reads."""
    model = _model().eval()
    x, _, y = _flat_batch()
    enc = model.noisy_encoder
    # The df head is zero-weighted at init (so the posterior opens on the
    # previous fixed t_5); give it a step's worth of weight first, otherwise
    # it is label-blind by construction.
    with torch.no_grad():
        enc.nu.weight.normal_(std=0.5)
        mu_a, lv_a, nu_a = enc(x, y)
        mu_b, lv_b, nu_b = enc(x, (y + 1) % 2)
    assert torch.equal(mu_a, mu_b)            # location: q(ẑ | x)
    assert not torch.allclose(lv_a, lv_b)     # scale reads the label
    assert not torch.allclose(nu_a, nu_b)     # df reads the label
    assert nu_a.shape == (x.shape[0],)        # one df per row
    assert (nu_a > 2.0).all() and (nu_a <= 30.0).all()


def test_the_df_head_opens_on_the_previous_fixed_df():
    """The learned head is biased so a fresh posterior is the old t_5."""
    enc = _model().eval().noisy_encoder
    x, _, y = _flat_batch()
    with torch.no_grad():
        _, _, nu = enc(x, y)
    assert torch.allclose(nu, torch.full_like(nu, 5.0), atol=1e-5)


def test_a_fixed_df_bundle_is_rejected_with_a_clear_error(tmp_path):
    """The df is now learned, so a bundle without the ν head cannot be loaded;
    the loader must say so rather than fail on an opaque missing key."""
    from scripts.lsnpc_ckpt import load_bundle, save_bundle

    model = _model()
    arch = {"build_kwargs": dict(
        x_dim=4, n_classes=2, latent_dim=3, hidden_dim=8, n_blocks=1, nu0=5.0,
        beta=1.0, image_data=False, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=False, head="concat")}
    path = save_bundle(tmp_path / "old.pt", model, arch=arch,
                       splits={}, run={}, val_err_h=0.4)
    bundle = torch.load(path, weights_only=False)
    for key in ("noisy_encoder.nu.weight", "noisy_encoder.nu.bias"):
        bundle["lsnpc_state"].pop(key)
    torch.save(bundle, path)

    with pytest.raises(ValueError, match="fixed-df posterior"):
        load_bundle(path)


def test_scratch_output_dir_carries_the_same_flag_tokens_as_the_bundle():
    """The two collided once: a seed-42 result.json was overwritten by a
    different arm of the same cell, because only the bundle name had tokens."""
    from scripts.image_lsnpc import _lsnpc_ckpt_name, _lsnpc_output_dir

    config = SimpleNamespace(lsnpc_img_size=32)
    args = SimpleNamespace(
        dataset="cifar10n", seed=42, epochs=3, beta=0.5, clean_set_size=2000,
        focal_gamma=0.0, focal_alpha=None, lsnpc_head="concat",
        lsnpc_embed_dim=128,
        correction_cond="none", shared_yhat_embed=False,
        correction_input="gated", eta=0.1, backbone="resnet50")
    default_name = _lsnpc_ckpt_name(config, args, True, True, "worse")
    assert _lsnpc_output_dir(config, args, True, True, "worse") == \
        "/tmp/image_" + default_name[len("lsnpc_"):-len(".pt")]

    args.lsnpc_head = "gate"
    args.lsnpc_embed_dim, args.correction_cond = 64, "x_yhat"
    name = _lsnpc_ckpt_name(config, args, True, True, "worse")
    directory = _lsnpc_output_dir(config, args, True, True, "worse")
    for token in ("_hgate", "_ed64", "_ccx_yhat"):
        assert token in name and token in directory, token


def test_gate_head_survives_a_bundle_round_trip(tmp_path):
    from scripts.lsnpc_ckpt import load_bundle, save_bundle

    # ``build_lsnpc`` uses the LSNPC default dropout (0.1) and the shim does
    # not carry it, so the round-trip model must be built the same way. x_dim
    # sizes the data decoder, so it has to match the arch written below.
    model = _model(head_type="gate", image_data=True, dropout=0.1,
                   x_dim=3 * 32 * 32)
    arch = {"build_kwargs": dict(
        x_dim=3 * 32 * 32, n_classes=2,
        latent_dim=3, hidden_dim=8, n_blocks=1, student_t_nu=5.0, nu0=5.0,
        beta=1.0, image_data=True, img_channels=3, img_size=32,
        encoder_backbone="conv", freeze_backbone=False, head="gate")}
    path = save_bundle(tmp_path / "bundle.pt", model, arch=arch,
                       splits={"a": None}, run={}, val_err_h=0.4)
    loaded = load_bundle(path)
    assert loaded["model"].label_head.head_type == "gate"
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded["model"].state_dict()[key]), key


# ── softened label impact: the dial between the two leak-break ends ──────

def test_correction_conditioning_arms_size_the_gate_input():
    x, _, y = _flat_batch()
    assert _model(correction_cond="none").correction_cond_vector(x, y) is None
    x_only = _model(correction_cond="x")
    assert x_only.correction_encoder.cond_dim == 4
    assert x_only.correction_cond_vector(x, y).shape == (8, 4)
    both = _model(correction_cond="x_yhat")
    assert both.correction_encoder.cond_dim == 4 + 2
    assert both.correction_cond_vector(x, y).shape == (8, 6)
    y_only = _model(correction_cond="yhat")
    assert y_only.correction_encoder.cond_dim == 0
    assert y_only.correction_cond_vector(x, y) is None


def test_unknown_correction_arm_is_rejected():
    with pytest.raises(ValueError, match="correction_cond"):
        _model(correction_cond="zhat_yhat")


def test_the_yhat_arm_replaces_the_x_latent_with_a_label_embedding():
    x, z_vae, y = _flat_batch()
    model = _model(correction_cond="yhat").eval()
    with torch.no_grad():
        here = model.blend_target(z_vae, y)
        other = model.blend_target(z_vae, 1 - y)
        ignores_x = model.blend_target(z_vae * 5.0, y)
    assert here.shape == z_vae.shape
    assert not torch.allclose(here, other)      # it reads y_hat
    assert torch.allclose(here, ignores_x)      # and no longer reads x
    # Every other arm still blends against the VAE latent of x.
    assert torch.allclose(
        _model(correction_cond="none").blend_target(z_vae, y), z_vae)


def test_the_yhat_arm_grades_its_blend_target_and_keeps_the_objective_finite():
    x, z_vae, y = _flat_batch()
    model = _model(correction_cond="yhat").train()
    out = _loss(model, x, z_vae, y)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0.0
    out["loss"].backward()
    grad = model.yhat_blend_target.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_the_yhat_arm_survives_the_inference_path():
    x, z_vae, y = _flat_batch()
    model = _model(correction_cond="yhat").eval()
    with torch.no_grad():
        log_w, z_samples = model.iw_log_weights(x, z_vae, y, K=2)
        out = model.forward(x, y, z_vae=z_vae, K=2)
    assert torch.isfinite(log_w).all()
    assert z_samples.shape == (2, 8, 3)
    assert out["corrected_logits"].shape == (8, 2)


def test_ckpt_name_records_the_conditioning_arm():
    from scripts.image_lsnpc import _lsnpc_ckpt_name

    config = SimpleNamespace(lsnpc_img_size=32)
    args = SimpleNamespace(
        dataset="cifar10n", seed=42, epochs=3, beta=0.5, clean_set_size=2000,
        focal_gamma=0.0, focal_alpha=None, lsnpc_head="concat",
        lsnpc_embed_dim=128, correction_cond="none",
        shared_yhat_embed=False, correction_input="gated",
        eta=0.1, backbone="resnet50")
    default_name = _lsnpc_ckpt_name(config, args, True, True)
    # Token forms carry the value, so they cannot collide with the dataset
    # name itself ("lsnpc_cifar10n" contains "_ci").
    assert "_ccnone" not in default_name and "_cigated" not in default_name

    args.correction_cond, args.correction_input = "yhat", "concat"
    name = _lsnpc_ckpt_name(config, args, True, True)
    for token in ("_ccyhat", "_ciconcat"):
        assert token in name, token


def test_the_config_rejects_a_bad_conditioning_arm():
    from experiments.arguments import ExperimentConfig

    bad_arm = ExperimentConfig()
    bad_arm.lsnpc_correction_cond = "zhat_yhat"
    with pytest.raises(ValueError, match="lsnpc_correction_cond"):
        bad_arm.validate()


# ── explicit shape channel: the label reaches scale and tail, never location ──
