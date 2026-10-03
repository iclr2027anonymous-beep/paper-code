"""The backbone's own norm layers follow its frozen/trainable flag.

There is one switch: ``freeze_backbone``. A frozen backbone keeps the
pretrained BatchNorm running statistics (BN in eval, statistics untouched); a
trainable backbone fine-tunes its norm layers with the rest of its weights.
An earlier iteration had a separate ``freeze_bn`` knob and an RMSNorm swap for
the ResNet trunk; both are gone, so these tests pin the coupling and the
absolutely-stock torchvision architecture.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.archs import ResNet50Encoder, build_image_encoder
from models.lsnpc import build_lsnpc


@pytest.fixture(autouse=True)
def offline_resnet_weights(monkeypatch):
    import torchvision.models as tv_models
    for name in ("resnet50", "resnet101"):
        original = getattr(tv_models, name)
        def build_without_download(*args, _original=original, **kwargs):
            kwargs["weights"] = None
            return _original(*args, **kwargs)
        monkeypatch.setattr(tv_models, name, build_without_download)


def _bns(encoder):
    return [m for m in encoder.net.modules() if isinstance(m, nn.BatchNorm2d)]


def test_resnet50_is_stock_torchvision_plus_heads():
    enc = ResNet50Encoder(8, 3, 32, pretrained=False, freeze_backbone=False)
    assert [n for n, _ in enc.named_children()] == ["net", "mu", "logvar"]
    assert enc.mu.in_features == 2048
    for name in ("trunk", "norm", "act", "head"):
        assert not hasattr(enc, name)
    assert not any("RMSNorm" in type(m).__name__ for m in enc.net.modules())


@pytest.mark.parametrize("freeze_backbone,frozen", [(True, True), (False, False)])
def test_batchnorm_mode_follows_freeze_backbone(freeze_backbone, frozen):
    enc = ResNet50Encoder(8, 3, 32, pretrained=False,
                          freeze_backbone=freeze_backbone)
    enc.train()
    bns = _bns(enc)
    assert bns, "the torchvision ResNet-50 must keep its BatchNorm layers"
    assert all(bn.training is not frozen for bn in bns)


def test_frozen_backbone_keeps_the_pretrained_running_statistics():
    enc = ResNet50Encoder(8, 3, 32, pretrained=False, freeze_backbone=True)
    enc.train()
    bn = _bns(enc)[0]
    # A non-trivial batch must not move the running statistics when frozen.
    before = bn.running_mean.clone()
    with torch.no_grad():
        bn(torch.randn(8, bn.num_features, 4, 4) * 3.0 + 1.0)
    assert torch.equal(bn.running_mean, before)


def test_trainable_backbone_updates_the_running_statistics():
    enc = ResNet50Encoder(8, 3, 32, pretrained=False, freeze_backbone=False)
    enc.train()
    bn = _bns(enc)[0]
    before = bn.running_mean.clone()
    with torch.no_grad():
        bn(torch.randn(8, bn.num_features, 4, 4) * 3.0 + 1.0)
    assert not torch.equal(bn.running_mean, before)


@pytest.mark.parametrize("backbone", ["resnet50", "resnet101"])
def test_build_image_encoder_forwards_freeze_backbone(backbone):
    frozen = build_image_encoder(backbone, 8, 3, 32, freeze_backbone=True)
    assert all(not p.requires_grad for p in frozen.net.parameters())
    trainable = build_image_encoder(backbone, 8, 3, 32, freeze_backbone=False)
    assert all(p.requires_grad for p in trainable.net.parameters())
    # The mu/logvar heads always train, frozen backbone or not.
    assert all(p.requires_grad for p in frozen.mu.parameters())
    assert all(p.requires_grad for p in frozen.logvar.parameters())


def test_build_lsnpc_couples_the_norm_mode_to_the_backbone():
    def shim(freeze_backbone):
        return SimpleNamespace(
            latent_dim=8, hidden_dim=16, n_blocks=2,
            lsnpc_nu0=2.0, lsnpc_beta=0.5, lsnpc_focal_gamma=0.0,
            lsnpc_focal_alpha=None, lsnpc_head="concat", lsnpc_embed_dim=128,
            lsnpc_correction_cond="none",
            lsnpc_image_data=True, lsnpc_img_channels=3, lsnpc_img_size=32,
            encoder_backbone="resnet50", freeze_backbone=freeze_backbone)

    for freeze_backbone in (True, False):
        model = build_lsnpc(shim(freeze_backbone), x_dim=48, n_classes=3)
        model.train()
        enc = model.noisy_encoder.image_encoder
        assert enc.freeze_bn is freeze_backbone
        assert all(bn.training is not freeze_backbone for bn in _bns(enc))


def test_config_has_no_freeze_bn_knob():
    from experiments.arguments import ExperimentConfig
    assert "freeze_bn" not in ExperimentConfig.__dataclass_fields__
