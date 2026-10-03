"""clean_set_size semantics: -1 = use all clean rows, 0 = no clean set
(unsupervised correction), >0 = cap; anything below -1 is rejected."""

import numpy as np
import pytest
import torch

from experiments.arguments import ExperimentConfig
from experiments.lsnpc_stage1 import _select_clean_set


class _ValSplit:
    """Minimal val_clean stand-in: .features / .targets torch tensors."""

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)


def _config(**overrides):
    base = dict(use_semi=True, seed=42, clean_set_size=-1)
    base.update(overrides)
    return ExperimentConfig(**base)


def _val_split(n_rows=120, n_feat=4):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(n_rows, n_feat)).astype(np.float32)
    targets = np.tile(np.array([0, 1]), n_rows // 2).astype(np.int64)
    return _ValSplit(torch.as_tensor(features), torch.as_tensor(targets))


def test_val_provided_minus_one_uses_all_val_rows():
    cfg = _config(clean_set_size=-1)
    val = _val_split()
    X_clean_set, y_clean, clean_idx, m = _select_clean_set(
        cfg, np.random.default_rng(cfg.seed), val_clean=val)
    assert clean_idx is None
    assert m == len(val)
    assert len(X_clean_set) == len(val)
    assert len(y_clean) == len(val)
    np.testing.assert_array_equal(X_clean_set, val.features.cpu().numpy())
    np.testing.assert_array_equal(y_clean, val.targets.cpu().numpy().ravel())


def test_val_provided_positive_cap_subsamples_val():
    cfg = _config(clean_set_size=20)
    val = _val_split()
    X_clean_set, y_clean, clean_idx, m = _select_clean_set(
        cfg, np.random.default_rng(cfg.seed), val_clean=val)
    assert clean_idx is None
    assert m == 20
    assert len(X_clean_set) == 20
    assert len(y_clean) == 20
    # Deterministic per seed and drawn from the val split only.
    X_again, _, _, m_again = _select_clean_set(
        cfg, np.random.default_rng(cfg.seed), val_clean=val)
    assert m_again == 20
    np.testing.assert_array_equal(X_again, X_clean_set)
    val_rows = {
        r.tobytes() for r in np.ascontiguousarray(val.features.cpu().numpy())
    }
    clean_rows = {r.tobytes() for r in np.ascontiguousarray(X_clean_set)}
    assert len(clean_rows) == 20
    assert clean_rows <= val_rows


def test_val_provided_cap_larger_than_split_uses_all():
    cfg = _config(clean_set_size=10_000)
    val = _val_split()
    X_clean_set, _, _, m = _select_clean_set(
        cfg, np.random.default_rng(cfg.seed), val_clean=val)
    assert m == len(val)
    assert len(X_clean_set) == len(val)


def test_training_drawn_minus_one_raises_clear_error():
    """-1 (use all) cannot be honoured when the clean set comes from train."""
    cfg = _config(clean_set_size=-1)
    rng = np.random.default_rng(cfg.seed)
    X_tr = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    y_clean_tr = np.zeros(100, dtype=np.int64)
    with pytest.raises(ValueError, match="use all"):
        _select_clean_set(cfg, rng, X_train=X_tr, y_train_clean=y_clean_tr)


def test_training_drawn_zero_yields_no_clean_rows():
    """0 = no clean set, and it is legal wherever -1 is not."""
    cfg = _config(clean_set_size=0)
    rng = np.random.default_rng(cfg.seed)
    X_tr = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    y_clean_tr = np.zeros(100, dtype=np.int64)
    X_clean_set, y_clean, clean_idx, m = _select_clean_set(
        cfg, rng, X_train=X_tr, y_train_clean=y_clean_tr)
    assert m == 0
    assert len(X_clean_set) == 0
    assert len(y_clean) == 0
    assert len(clean_idx) == 0


def test_training_drawn_positive_cap_draws_from_train():
    cfg = _config(clean_set_size=10)
    rng = np.random.default_rng(cfg.seed)
    X_tr = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    y_clean_tr = (np.arange(100) % 2).astype(np.int64)
    X_clean_set, y_clean, clean_idx, m = _select_clean_set(
        cfg, rng, X_train=X_tr, y_train_clean=y_clean_tr)
    assert m == 10
    assert len(clean_idx) == 10
    assert len(set(clean_idx.tolist())) == 10  # no duplicated rows
    np.testing.assert_array_equal(X_clean_set, X_tr[clean_idx])
    np.testing.assert_array_equal(y_clean, y_clean_tr[clean_idx])


def test_config_validation_accepts_use_all_none_and_positive():
    ExperimentConfig(use_semi=True, clean_set_size=-1).validate()
    ExperimentConfig(use_semi=True, clean_set_size=0).validate()
    ExperimentConfig(use_semi=True, clean_set_size=50).validate()


def test_config_validation_rejects_below_minus_one():
    with pytest.raises(ValueError, match="clean_set_size"):
        ExperimentConfig(use_semi=True, clean_set_size=-2).validate()


def test_val_provided_zero_yields_no_clean_rows():
    """0 = no clean set even when a held-out clean split exists."""
    cfg = _config(clean_set_size=0)
    val = _val_split()
    X_clean_set, y_clean, clean_idx, m = _select_clean_set(
        cfg, np.random.default_rng(cfg.seed), val_clean=val)
    assert m == 0
    assert len(X_clean_set) == 0
    assert len(y_clean) == 0
