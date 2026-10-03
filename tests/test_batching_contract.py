"""The one-loader contract: `utils.batching` builds it, `accel.prepare` places it.

Reproducibility, not style, is what these tests protect. The protocol scorers
read the global CPU RNG stream that `torch.utils.data` advances once per
`iter()`, so the wrapper that carries batches to the device must not add,
remove or reorder a single draw.
"""
import numpy as np
import torch

from utils.batching import (
    _DeviceLoader, batched_predict, make_loader, place_loader,
    place_loader_local, tensor_loader,
)

N_ROWS, D = 23, 6  # a short trailing batch at bs=5


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N_ROWS, D, generator=g), torch.randint(0, 3, (N_ROWS,), generator=g)


def test_factory_disables_persistent_workers():
    X, y = _data()
    dl = tensor_loader(X, y, batch_size=5, shuffle=True, num_workers=0)
    assert dl.persistent_workers is False


def test_factory_leaves_pinning_off_unless_asked():
    X, y = _data()
    dl = tensor_loader(X, y, batch_size=5)
    assert dl.pin_memory is False
    assert tensor_loader(X, y, batch_size=5, pin_memory=True).pin_memory is True


def test_putting_batches_on_the_device_does_not_shift_the_global_rng():
    """`place_loader` must leave the RNG stream exactly where the loader did."""
    X, y = _data()

    torch.manual_seed(7)
    ref_batches = [b for b in tensor_loader(X, y, batch_size=5, shuffle=True)]
    ref_state = torch.random.get_rng_state()

    torch.manual_seed(7)
    placed = place_loader(None, tensor_loader(X, y, batch_size=5, shuffle=True), "cpu")
    got_batches = [b for b in placed]

    assert torch.equal(ref_state, torch.random.get_rng_state())
    assert len(got_batches) == len(ref_batches)
    for (xb, yb), (rxb, ryb) in zip(got_batches, ref_batches):
        assert torch.equal(xb, rxb) and torch.equal(yb, ryb)


def test_tensor_loader_and_make_loader_agree_row_for_row():
    X, y = _data()
    from torch.utils.data import TensorDataset

    a = list(tensor_loader(X, y, batch_size=7, shuffle=False))
    b = list(make_loader(TensorDataset(X, y), batch_size=7, shuffle=False))
    assert len(a) == len(b) == 4  # 3 x 7 + 1 x 2
    for (x1, y1), (x2, y2) in zip(a, b):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_place_loader_leaves_none_alone():
    assert place_loader(None, None, "cpu") is None


def test_device_loader_delegates_everything_else_to_the_inner_loader():
    X, y = _data()
    dl = tensor_loader(X, y, batch_size=5, drop_last=False)
    wrapped = _DeviceLoader(dl, "cpu")
    assert len(wrapped) == len(dl) == 5
    assert wrapped.batch_size == dl.batch_size


def test_batched_predict_matches_the_single_shot_reference():
    X, _ = _data()

    class M(torch.nn.Module):
        def forward(self, x):
            return x.square()

    for bs in (1, 7, N_ROWS):
        got = batched_predict(M(), X.numpy(), batch_size=bs, device="cpu")
        assert np.array_equal(got, (X.numpy() ** 2))


def test_place_loader_local_never_shards_a_whole_dataset_caller():
    """``accel.prepare`` shards a map-style loader across processes, and a
    caller that indexes rows by position or rebuilds the input order cannot
    survive that. Asserted against an Accelerator stand-in that really does
    shard, so this fails loudly rather than passing vacuously if the
    underlying behaviour ever changes."""
    from accelerate.data_loader import prepare_data_loader

    X, y = _data()
    sharding_accel = prepare_data_loader(
        tensor_loader(X, y, batch_size=5, shuffle=False),
        num_processes=2, process_index=0, put_on_device=False)
    assert sum(b[0].shape[0] for b in sharding_accel) < N_ROWS  # the premise

    placed = place_loader_local(
        tensor_loader(X, y, batch_size=5, shuffle=False), "cpu")
    rows = torch.cat([b[0] for b in placed])
    assert torch.equal(rows, X)
    assert torch.equal(torch.cat([b[1] for b in placed]), y)
