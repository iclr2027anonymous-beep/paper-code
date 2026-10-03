"""The pretrained-backbone pixel transform.

The pipelines carry images as CIFAR-normalized tensors (``(raw/255 - CIFAR_MEAN)
/ CIFAR_STD``) at the dataset's own resolution. A pretrained ImageNet backbone
was trained on ``(raw/255 - IMAGENET_MEAN) / IMAGENET_STD`` at 224x224, so
something has to convert between the two. That conversion lives here so the
encoders that apply it share one definition.

The conversion is invariant to bf16 autocast -- every op in it runs in fp32
either way (measured: bit-identical output under
``torch.autocast("cuda", bfloat16)``) and to the device (measured:
bit-identical CPU vs CUDA on fp32 and bf16 input).

It is applied *in the model*, on the batch, not in the data pipeline: measured
on the frozen pixel step it costs 0.18 ms per 64-image batch on the GPU against
11.6 ms on the CPU, so a loader-side transform is 3.4 s/epoch slower and, in a
DataLoader worker (pinned to one intra-op thread), not even bit-identical. See
the implementation notes 5.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Square input side the pretrained backbones expect.
PRETRAINED_INPUT_SIZE = 224

# Canonical normalization constants (PyTorch convention). The pipeline's
# loaders normalize with the CIFAR pair (data_process/cifar10n._normalize).
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def pretrained_pixel_transform(x: torch.Tensor,
                               size: int = PRETRAINED_INPUT_SIZE,
                               ) -> torch.Tensor:
    """CIFAR-normalized pixels -> ImageNet-normalized ``size``x``size`` pixels.

    Accepts NHWC or NCHW, returns NCHW (the layout the backbones' first
    convolution wants): (1) denormalize CIFAR back to raw [0, 1], (2) resize,
    (3) apply ImageNet normalization.
    """
    if x.dim() != 4:
        raise ValueError(
            f"expected a 4-D batched image tensor, got shape {tuple(x.shape)}")
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    dev = x.device
    cm = torch.tensor(_CIFAR_MEAN, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    cs = torch.tensor(_CIFAR_STD, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    im = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    ist = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    x = x * cs + cm                       # CIFAR-normalized -> raw [0, 1]
    x = F.interpolate(x, size=(size, size), mode="bilinear",
                      align_corners=False)
    return (x - im) / ist
