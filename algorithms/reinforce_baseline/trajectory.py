from __future__ import annotations
import numpy as np
from dataclasses import dataclass

MIN_STEP_OUT_LEN = 3
STEP_OUT_WITH_INFO_LEN = 4

@dataclass(slots=True)
class EpisodeStep:
    """Stores everything needed to update policy and baseline."""
    policy_features: np.ndarray
    action_index: int
    reward: float
    value_features: np.ndarray | None

class TrajectoryRunner:
    """Generates complete episodes given a policy and baseline."""

    def __init__(self, env):
        self.env = env

    def rollout(self, policy, baseline, max_steps):
        """Generates a complete episode given a policy and baseline"""
        state = self.env.reset()
        steps = []
        for x in range(max_steps):
            idx, action, policy_features = policy.sample(state)
            value_features = baseline.features(state)
            next_state, reward, done = self._step(state, action)
            steps.append(
                EpisodeStep(
                    policy_features=policy_features,
                    action_index=idx,
                    reward=float(reward),
                    value_features=value_features,
                )
            )
            state = next_state
            if done:
                break
        return steps

    def _step(self, state, action):
        """Takes a step in the environment"""
        if hasattr(self.env, "sample_transition"):
            next_state, reward, done = self.env.sample_transition(state, action)
            return next_state, float(reward), bool(done)

        try:
            step_out = self.env.step(action)
        except TypeError:
            step_out = self.env.step(state, action)

        if len(step_out) == MIN_STEP_OUT_LEN:
            next_state, reward, done = step_out
        elif len(step_out) >= STEP_OUT_WITH_INFO_LEN:
            next_state, reward, done = step_out[0], step_out[1], step_out[2]

        return next_state, float(reward), bool(done)


