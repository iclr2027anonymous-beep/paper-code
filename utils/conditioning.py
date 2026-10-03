"""Conditioning-vector construction shared by training and evaluation."""

from __future__ import annotations

import numpy as np


def build_predictive_conditioning(
    clf, X: np.ndarray, mode: str = "posterior"
) -> np.ndarray:
    """Return posterior or argmax one-hot generative conditioning."""
    probabilities = np.asarray(clf.predict_proba(X), dtype=np.float32)
    probabilities /= np.clip(
        probabilities.sum(axis=1, keepdims=True), 1e-8, None
    )
    if mode == "posterior":
        return probabilities
    if mode == "hard":
        hard = np.zeros_like(probabilities)
        hard[np.arange(len(hard)), probabilities.argmax(axis=1)] = 1.0
        return hard
    raise ValueError(f"Unknown conditioning mode: {mode}")
