import numpy as np
import pytest
import torch
from torch import nn

from types import SimpleNamespace

from experiments.lsnpc_stage1 import _classification_correction_diagnostics
from trainers.lsnpc import _backbone_liveness


def test_correction_diagnostics_report_error_calibration_and_brier():
    y_true = np.array([0, 1, 1, 0])
    noisy_prob = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7],
    ])
    corrected_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.7, 0.3],
    ])

    metrics = _classification_correction_diagnostics(
        y_true, noisy_prob, corrected_prob)

    assert metrics["diagnostic_test_size"] == 4
    assert metrics["noisy_conditioning_error"] == pytest.approx(0.5)
    assert metrics["corrected_conditioning_error"] == pytest.approx(0.0)
    assert metrics["conditioning_error_reduction"] == pytest.approx(0.5)
    assert metrics["corrected_ece"] < metrics["noisy_ece"]
    assert metrics["corrected_brier"] < metrics["noisy_brier"]
    assert metrics["corrected_confusion_matrix"] == [[2, 0], [0, 2]]


def test_correction_diagnostics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        _classification_correction_diagnostics(
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            np.array([[0.8, 0.2]]),
        )


def _stub_lsnpc(enc, image_data=True):
    return SimpleNamespace(noisy_encoder=SimpleNamespace(image_encoder=enc),
                           image_data=image_data)


class _FeatureStub(nn.Module):
    """Minimal image encoder exposing ``backbone_feature`` for the guard."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def backbone_feature(self, x):
        return self._fn(x)


def test_backbone_liveness_skips_text_runs_and_encoders_without_a_backbone():
    x = torch.zeros(2, 3, 8, 8)
    assert _backbone_liveness(_stub_lsnpc(object(), image_data=False), x,
                              "t") is None
    assert _backbone_liveness(_stub_lsnpc(object()), x, "t") is None


def test_backbone_liveness_aborts_on_a_constant_or_non_finite_feature():
    x = torch.zeros(2, 3, 8, 8)
    flat = _FeatureStub(lambda _x: torch.zeros(_x.size(0), 16))
    with pytest.raises(RuntimeError, match="DEAD"):
        _backbone_liveness(_stub_lsnpc(flat), x, "t")
    inf = _FeatureStub(
        lambda _x: torch.full((_x.size(0), 16), float("inf")))
    with pytest.raises(RuntimeError, match="non-finite"):
        _backbone_liveness(_stub_lsnpc(inf), x, "t")


def test_backbone_liveness_reports_the_feature_spread():
    x = torch.zeros(2, 3, 8, 8)
    enc = _FeatureStub(lambda _x: torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]))
    assert _backbone_liveness(_stub_lsnpc(enc), x, "t") == pytest.approx(
        2.8284, rel=1e-3)
