"""Shared MLP builder: FC → RMSNorm → activation → Dropout blocks."""

import torch.nn as nn


def build_mlp(dims: list[int], dropout: float = 0.1,
              activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    """Build an MLP as a sequence of FC → RMSNorm → activation → Dropout blocks.

    ``dims[0]`` is the input dimension, ``dims[-1]`` the output dimension.
    Every intermediate block is ``Linear → RMSNorm → activation → Dropout``.
    The final layer is a plain ``Linear`` with no norm/activation/dropout.
    A ``dims`` of length 2 keeps only that final ``Linear``: the loop below
    builds ``len(dims) - 2`` blocks, so ``build_mlp([48, 16])`` is a bare
    linear map, not a one-block MLP.

    Weights use PyTorch's default ``nn.Linear`` initialisation.  Custom init
    is deliberately *off*: hand-rolled narrow final projections (e.g.
    ``N(0, 0.01)``) were found to collapse signal through deep signal-path
    MLPs.  PyTorch's default (uniform ``±1/√fan_in``)
    preserves signal through every layer.

    Example::

        net = build_mlp([32, 128, 64, 10], dropout=0.1)
        #   Linear(32, 128) → RMSNorm(128) → GELU → Dropout(0.1)
        #   Linear(128, 64)  → RMSNorm(64)  → GELU → Dropout(0.1)
        #   Linear(64, 10)
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        in_d, out_d = dims[i], dims[i + 1]
        layers.append(nn.Linear(in_d, out_d))
        layers.append(nn.RMSNorm(out_d))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    # Final projection (no norm/activation/dropout)
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)
