"""Regression test for ``--lsnpc-backbone-lr-scale`` on the LSNPC trainer.

LSNPC keeps two independent image backbones (the posterior encoder's and the
label head's) and has no ``image_encoder`` attribute of its own.  Looking the
backbone up on the model alone therefore produced an EMPTY scaled group, and
every parameter -- backbones included -- trained at the head LR.  The knob was
silent about it, which is how the H2 ablation in the implementation notes
came to report that a 0.02x backbone LR changes nothing.
"""
import pytest
import torch
import torch.nn as nn

from trainers.lsnpc import LSNPCTrainer


class _TrainerStub:
    """Just enough of LSNPCTrainer for the unbound ``_param_groups``."""


def _stub_model() -> nn.Module:
    model = nn.Module()
    model.noisy_encoder = nn.Module()
    model.noisy_encoder.image_encoder = nn.Module()
    model.noisy_encoder.image_encoder.net = nn.Sequential(nn.Linear(2, 2))
    model.label_head = nn.Module()
    model.label_head.image_encoder = nn.Module()
    model.label_head.image_encoder.net = nn.Sequential(nn.Linear(3, 3))
    model.head = nn.Linear(4, 4)
    return model


def test_backbone_lr_scale_puts_both_backbones_in_the_scaled_group():
    trainer = _TrainerStub()
    trainer.model = _stub_model()
    trainer._bb_lr_scale = 0.02

    rest, backbone = LSNPCTrainer._param_groups(trainer, lr=1e-3)

    assert backbone["lr"] == pytest.approx(2e-5)
    assert rest["lr"] == pytest.approx(1e-3)
    assert len(backbone["params"]) == 4  # 2 + 2 backbone tensors
    assert {id(p) for p in backbone["params"]} == {
        id(p) for p in trainer.model.noisy_encoder.image_encoder.net.parameters()
    } | {
        id(p) for p in trainer.model.label_head.image_encoder.net.parameters()
    }
    assert id(next(trainer.model.head.parameters())) in {
        id(p) for p in rest["params"]}


def test_backbone_lr_scale_one_is_inert():
    trainer = _TrainerStub()
    trainer.model = _stub_model()
    trainer._bb_lr_scale = 1.0

    groups = LSNPCTrainer._param_groups(trainer, lr=1e-3)

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert len(groups[0]["params"]) == len(list(trainer.model.parameters()))
