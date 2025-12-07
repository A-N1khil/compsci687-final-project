from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from .baseline import BaselineModule
from .policy import PolicyModule
from .trajectory import TrajectoryRunner

FeatureFn = Callable[[Any], np.ndarray]


@dataclass(slots=True)
class ReinforceAgentConfig:
    """Hyper-parameters controlling REINFORCE learning behaviour."""
    gamma: float = 0.99
    alpha_theta: float = 0.01
    alpha_w: float = 0.05
    use_baseline: bool = True
    seed: int | None = None

@dataclass(slots=True)
class EpisodeStats:
    """Tracks total return and the number of steps in an episode."""
    episode_return: float
    length: int

class ReinforceBaselineAgent:
    """Episodic REINFORCE with an optional learned baseline."""

    def __init__(self, action_space, policy_feature_fn, value_feature_fn = None, config = None):
        """Initializes the ReinforceBaselineAgent"""
        self.config = config if config is not None else ReinforceAgentConfig()
        value_fn = value_feature_fn or policy_feature_fn

        self.policy = PolicyModule(action_space, policy_feature_fn, seed=self.config.seed)
        self.baseline = BaselineModule(value_fn,alpha_w=self.config.alpha_w, enabled=self.config.use_baseline)

    def train(self, env, episodes, max_steps):
        """Trains the ReinforceBaselineAgent"""
        runner = TrajectoryRunner(env)
        stats = []
        for _ in range(episodes):
            episode = runner.rollout(self.policy, self.baseline, max_steps)
            ep_return = self._update_from_episode(episode)
            stats.append(EpisodeStats(episode_return=ep_return, length=len(episode)))
        return stats

    def act(self, state):
        i, action, features = self.policy.sample(state)
        return action

    def _update_from_episode(self, episode):
        """Updates the ReinforceBaselineAgent from an episode"""
        returns = self._discounted_returns([step.reward for step in episode])
        episode_return = returns[0] if returns else 0.0

        for t, (step, G_t) in enumerate(zip(episode, returns)):
            baseline_value = self.baseline.value(step.value_features)
            delta = G_t - baseline_value
            self.baseline.update(delta, step.value_features)
            probs = self.policy.probs(step.policy_features)
            grad_log_pi = self.policy.grad_log_pi(probs, step.action_index, step.policy_features)
            step_size = self.config.alpha_theta * (self.config.gamma**t) * delta
            self.policy.apply_gradient(grad_log_pi, step_size)

        return episode_return

    def _discounted_returns(self, rewards):
        """Calculates the discounted returns"""
        G = 0.0
        returns = []
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.append(G)
        returns.reverse()
        return returns
