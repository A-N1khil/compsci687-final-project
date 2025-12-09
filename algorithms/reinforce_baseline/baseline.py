from __future__ import annotations
from typing import Any, Callable
import numpy as np

FeatureFn = Callable[[Any], np.ndarray]

class BaselineModule:
    """Maintains baseline weights"""

    def __init__(self, feature_fn, *, alpha_w, enabled = True):
        """Initializes the baseline module"""
        self.feature_fn = feature_fn
        self.alpha_w = alpha_w
        self.enabled = enabled
        self.w = None

    def features(self, state):
        """Extract features from the state"""
        if not self.enabled:
            return None
        arr = np.asarray(self.feature_fn(state), dtype=np.float64)
        return arr

    def value(self, features):
        """Calculates the value of the state"""
        if not self.enabled or features is None:
            return 0.0
        self._ensure_params(features.shape[0])
        assert self.w is not None
        return float(self.w @ features)

    def update(self, delta, features):
        """Updates the baseline weights"""
        if not self.enabled or features is None:
            return
        self._ensure_params(features.shape[0])
        assert self.w is not None
        self.w += self.alpha_w * delta * features

    def _ensure_params(self, feature_dim):
        """Ensures the baseline weights are initialized"""
        if self.w is None:
            self.w = np.zeros(feature_dim, dtype=np.float64)
        elif self.w.shape[0] != feature_dim:
            msg = "Value feature dimension changed during training"
            raise ValueError(msg)


