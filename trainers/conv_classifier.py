"""CNN classifier for image data — wraps ConvEncoder + linear head."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.archs import ConvEncoder
from utils.batching import batched_predict, place_loader, tensor_loader

log = logging.getLogger(__name__)


class ConvClassifier(nn.Module):
    """CNN classifier for 96x96 or 32x32 images, using ConvEncoder backbone."""

    def __init__(self, img_channels=3, img_size=96, n_classes=2, latent_dim=128):
        super().__init__()
        self.img_h = img_size
        self.img_w = img_size
        self.img_c = img_channels
        self.encoder = ConvEncoder(latent_dim=latent_dim,
                                   img_channels=img_channels,
                                   img_size=img_size)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        # Handle both 4D NHWC and 2D flattened inputs
        if x.ndim == 2:
            channel = self.img_c
            total = x.shape[1]
            side = int(round((total / channel) ** 0.5))
            if side * side * channel == total:
                x = x.reshape(-1, side, side, channel)
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        elif x.ndim == 4 and x.shape[1] in (1, 3):
            pass  # already NCHW
        mu, _ = self.encoder(x)
        return self.classifier(mu)

    def predict(self, x_np: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.eval()
        return batched_predict(
            self, x_np, fn=lambda m, xb: m(xb).argmax(dim=1),
            batch_size=batch_size,
        )

    def predict_proba(self, x_np: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.eval()
        return batched_predict(
            self, x_np, fn=lambda m, xb: torch.softmax(m(xb), dim=1),
            batch_size=batch_size,
        )

    def score(self, x_np: np.ndarray, y_np: np.ndarray) -> float:
        return (self.predict(x_np) == y_np).mean()


def train_conv_classifier(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    img_channels=3, img_size=96, n_classes=2,
    n_epochs: int = 100, batch_size: int = 64,
    lr: float = 1e-3, device: str = "cpu",
    accel: Any = None, num_workers: int = 0,
) -> ConvClassifier:
    """Train the ConvClassifier head. Batches arrive device-placed via
    ``place_loader``. ``num_workers`` defaults to 0 because ``X_train`` is an
    in-memory tensor; raise it only for real per-sample work.
    """
    model = ConvClassifier(img_channels=img_channels, img_size=img_size,
                           n_classes=n_classes)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dl = place_loader(
        accel,
        tensor_loader(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        ),
        device,
    )
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for xb, yb in train_dl:
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            val_acc = model.score(X_val, y_val)
            log.info(f"  ConvCLF epoch {epoch + 1}/{n_epochs}: loss={np.mean(losses):.4f} val_acc={val_acc:.3f}")

    return model
