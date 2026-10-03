import numpy as np
import pytest


@pytest.mark.parametrize(
    ("n_train", "clean_size", "expected_val", "expected_model_train"),
    [
        (7000, 20, 500, 6480),
        (700, 20, 70, 610),
        (700, 200, 70, 430),
    ],
)
def test_lsnpc_split_budget_does_not_starve_small_datasets(
    n_train, clean_size, expected_val, expected_model_train,
):
    remaining = n_train - clean_size
    target_val = max(1, int(np.floor(0.10 * n_train)))
    n_val = min(500, target_val, remaining)

    assert n_val == expected_val
    assert remaining - n_val == expected_model_train
