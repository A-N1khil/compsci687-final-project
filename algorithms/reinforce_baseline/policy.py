from __future__ import annotations
from typing import Any, Callable
import numpy as np

FeatureFn = Callable[[Any], np.ndarray]

class PolicyModule:
    """Encapsulates feature extraction, sampling, and gradients."""
    def __init__(self, action_space, feature_fn, *, seed = None):
        self.actions = list(action_space)
        self.feature_fn = feature_fn
        self.rng = np.random.default_rng(seed)
        self.theta = None

    def features(self, state):
        """Extract features from the state"""
        arr = np.asarray(self.feature_fn(state), dtype=np.float64)
        return arr

    def sample(self, state):
        """Samples an action from the policy"""
        features = self.features(state)
        probs = self.probs(features)
        i = int(self.rng.choice(len(self.actions), p=probs))
        return i, self.actions[i], features

    def probs(self, features):
        """Calculates the probabilities of the actions"""
        logits = self.logits(features)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def logits(self, features):
        """Calculates the logits of the actions"""
        self._ensure_params(features.shape[0])
        assert self.theta is not None
        return self.theta @ features

    def grad_log_pi(self, probs, action_idx, features) :
        """Calculates the gradient of the log probability of the action"""
        one_hot = np.zeros_like(probs)
        one_hot[action_idx] = 1.0
        diff = one_hot - probs
        return diff[:, None] * features[None, :]

    def apply_gradient(self, grad, scale):
        """Applies the gradient to the policy parameters"""
        self._ensure_params(grad.shape[1])
        assert self.theta is not None
        self.theta += scale * grad

    def _ensure_params(self, feature_dim):
        """Ensures the policy parameters are initialized"""
        if self.theta is None:
            self.theta = np.zeros((len(self.actions), feature_dim), dtype=np.float64)
        elif self.theta.shape[1] != feature_dim:
            msg = "Policy feature dimension changed during training"
            raise ValueError(msg)



