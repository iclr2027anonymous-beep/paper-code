"""Predictive conditioning (posterior c_soft) contract tests."""
import unittest

import numpy as np

from utils.conditioning import build_predictive_conditioning


class _PosteriorClassifier:
    """2-feature classifier with hand-computable probabilities."""

    def predict_proba(self, X):
        values = np.asarray(X, dtype=np.float32)[:, :1]
        return np.concatenate([0.2 + values, 0.8 - values], axis=1)


class PredictiveConditioningTests(unittest.TestCase):
    def test_predictive_conditioning_uses_classifier_probabilities(self):
        X = np.array([[0.1], [0.3]], dtype=np.float32)
        posterior = build_predictive_conditioning(_PosteriorClassifier(), X)
        expected = np.array([[0.3, 0.7], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(posterior, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
