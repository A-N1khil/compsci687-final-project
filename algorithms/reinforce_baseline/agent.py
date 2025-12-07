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
    """Tracks episode-level metrics collected during training."""
    episode_return: float
    length: int
    success: bool
    positive_rewards: int
    penalty_rewards: int
    baseline_mse: float

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
        episode_logs = []

        def _to_serializable(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, set):
                return sorted(value)
            return value

        for episode_idx in range(episodes):
            episode = runner.rollout(self.policy, self.baseline, max_steps)
            ep_return, baseline_mse = self._update_from_episode(episode)
            positive_rewards = sum(1 for step in episode if step.reward > 0)
            penalty_rewards = sum(1 for step in episode if step.reward < 0)
            success = bool(episode) and bool(episode[-1].done)
            stats.append(
                EpisodeStats(
                    episode_return=ep_return,
                    length=len(episode),
                    success=success,
                    positive_rewards=positive_rewards,
                    penalty_rewards=penalty_rewards,
                    baseline_mse=baseline_mse,
                )
            )
            episode_logs.append(
                {
                    "episode": episode_idx + 1,
                    "length": len(episode),
                    "return": ep_return,
                    "success": success,
                    "positive_rewards": positive_rewards,
                    "penalty_rewards": penalty_rewards,
                    "baseline_mse": baseline_mse,
                    "steps": [
                        {
                            "t": t,
                            "state": _to_serializable(step.state),
                            "action": _to_serializable(step.action),
                            "reward": float(step.reward),
                            "next_state": _to_serializable(step.next_state),
                            "done": bool(step.done),
                        }
                        for t, step in enumerate(episode)
                    ],
                }
            )
        return stats, episode_logs

    def act(self, state):
        i, action, features = self.policy.sample(state)
        return action

    def _update_from_episode(self, episode):
        """Updates the ReinforceBaselineAgent from an episode"""
        returns = self._discounted_returns([step.reward for step in episode])
        episode_return = returns[0] if returns else 0.0

        squared_error_sum = 0.0
        sample_count = 0

        for t, (step, G_t) in enumerate(zip(episode, returns)):
            baseline_value = self.baseline.value(step.value_features)
            delta = G_t - baseline_value
            squared_error_sum += float(delta * delta)
            sample_count += 1
            self.baseline.update(delta, step.value_features)
            probs = self.policy.probs(step.policy_features)
            grad_log_pi = self.policy.grad_log_pi(probs, step.action_index, step.policy_features)
            step_size = self.config.alpha_theta * (self.config.gamma**t) * delta
            self.policy.apply_gradient(grad_log_pi, step_size)

        baseline_mse = squared_error_sum / sample_count if sample_count else 0.0

        return episode_return, baseline_mse

    def _discounted_returns(self, rewards):
        """Calculates the discounted returns"""
        G = 0.0
        returns = []
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.append(G)
        returns.reverse()
        return returns
