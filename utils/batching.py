"""One way to build a batch loader, and one way to get it onto the device.

Every loader is built by ``make_loader`` and placed by ``accel.prepare``, which
is ``DataLoaderShard`` under a real ``Accelerator`` and ``_DeviceLoader`` under
the ``_NullAccel`` shim. Callers never move a batch themselves.

``place_loader_local`` is the same placement without the distributed sampler:
a real ``Accelerator`` shards a prepared loader across processes, so a caller
that needs every row must not go through it.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _default_pin_memory() -> bool:
    """Pinning is off by default.

    Two reasons it does not follow CUDA availability any more. First, pinning a
    tensor that is already on the GPU raises ``cannot pin 'torch.cuda.FloatTensor'``
    (it broke the featuriser's loader path). Second, the transfer is never the
    bottleneck for these loaders -- the GPU forward is -- so pinning bought
    nothing measurable. An explicit ``pin_memory=True`` is still honoured.
    """
    return False


def make_loader(dataset, *, batch_size: int, shuffle: bool = False,
                collate_fn: Callable | None = None,
                num_workers: int = 0, pin_memory: bool | None = None,
                drop_last: bool = False,
                generator: torch.Generator | None = None) -> DataLoader:
    """The single loader constructor.

    ``pin_memory`` is off unless asked for (see ``_default_pin_memory``).
    ``persistent_workers`` is always off: a persistent loader skips the
    per-``iter()`` base seed drawn from the global RNG, so re-iterating it
    desynchronises every later draw. Use ``num_workers=0`` unless
    ``__getitem__`` does real per-sample work.
    """
    kwargs = {}
    if generator is not None:
        kwargs["generator"] = generator
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=int(num_workers), collate_fn=collate_fn,
        pin_memory=_default_pin_memory() if pin_memory is None else bool(pin_memory),
        persistent_workers=False, **kwargs,
    )


def tensor_loader(*columns, batch_size: int, shuffle: bool = False,
                  collate_fn: Callable | None = None,
                  num_workers: int = 0, pin_memory: bool | None = None,
                  drop_last: bool = False,
                  generator: torch.Generator | None = None) -> DataLoader:
    """``make_loader`` over a ``TensorDataset`` of whole-sample columns."""
    return make_loader(
        TensorDataset(*columns), batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last, generator=generator)


def move_batch(batch, device) -> Iterable:
    """Every tensor in a loader batch, moved to ``device`` (nesting preserved)."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    return batch


class _DeviceLoader:
    """The no-``Accelerator`` counterpart of ``DataLoaderShard``.

    Same contract, without the distributed machinery. ``__iter__`` delegates to
    the inner loader so the per-``iter()`` RNG draw is unchanged.
    """

    def __init__(self, loader: DataLoader, device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            yield move_batch(batch, self.device)

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def place_loader(accel_like, loader: DataLoader, device):
    """``accel.prepare(loader)``, building the shim when ``accel_like`` is None.

    Under a real ``Accelerator`` with ``num_processes > 1`` this SHARDS a
    map-style loader (one rank-local slice of the batches per process), which
    is what training wants and what any caller assembling a whole-array result
    in row order must not have: pass ``accel_like=None`` there to get device
    placement without the distributed sampler.
    """
    if loader is None:
        return None
    if accel_like is None:
        from trainers.base import _NullAccel  # local: avoids an import cycle

        accel_like = _NullAccel(device)
    return accel_like.prepare(loader)


def place_loader_local(loader: DataLoader, device):
    """``place_loader`` for a caller that needs EVERY row, in input order.

    ``accel.prepare`` shards a map-style loader across processes (one
    rank-local slice per process), which is right for a training loop and
    wrong for every caller that concatenates its batches back into the
    caller's row order or indexes rows by position: the shard is a valid
    tensor of the wrong rows, so the failure is silent. This is the
    device-placement-only path; ``place_loader`` stays the sharding one.
    """
    return place_loader(None, loader, device)


def batched_predict(
    model: torch.nn.Module,
    X,
    batch_size: int = 256,
    device=None,
    fn: Callable | None = None,
) -> np.ndarray:
    """Run ``fn(model, batch)`` over ``X`` in ``shuffle=False`` batches.

    Returns the per-batch numpy outputs stacked on axis 0. ``fn`` defaults to
    calling the model. The model's train/eval mode is left to the caller.
    """
    if device is None:
        device = next(model.parameters()).device
    dl = tensor_loader(torch.as_tensor(X, dtype=torch.float32),
                       batch_size=batch_size, pin_memory=False)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in _DeviceLoader(dl, device):
            out = fn(model, xb) if fn is not None else model(xb)
            parts.append(out.cpu().numpy())
    return np.concatenate(parts, axis=0)
