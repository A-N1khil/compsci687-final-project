import gymnasium as gym
import numpy as np
from mdp.mdp_base import MDPBase


class MountainCarContinuousMDP(MDPBase):
    """
    MDP wrapper for MountainCarContinuous-v0 using Gymnasium.
    This follows the same interface pattern as the discrete MountainCarMDP.
    """

    def __init__(self, seed=None):
        # creating the continuous MountainCar environment
        self.env = gym.make("MountainCarContinuous-v0")

        # state: (position, velocity)
        self.state_space = self.env.observation_space

        # action: 1D continuous throttle in [low, high]
        self.action_space = self.env.action_space

        # goal position from the underlying env if available
        self.goal_pos = getattr(self.env.unwrapped, "goal_position", 0.45)

        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        """Reset environment and return initial state as float32 numpy array."""
        state, _ = self.env.reset()
        return np.array(state, dtype=np.float32)

    def get_state_space(self):
        """Return the continuous state space."""
        return self.state_space

    def get_action_space(self):
        """Return the continuous action space."""
        return self.action_space

    def step(self, action):
        """
        Take one step in the environment.

        action can be a scalar or a small numpy array; we convert it to
        the shape the environment expects.
        """
        # handle scalar vs array actions
        if np.isscalar(action) or np.shape(action) == ():
            action_to_env = np.array([action], dtype=np.float32)
        else:
            action_to_env = np.array(action, dtype=np.float32)

        next_state, reward, terminated, truncated, info = self.env.step(action_to_env)
        done = terminated or truncated

        return np.array(next_state, dtype=np.float32), reward, done, info

    def is_terminal(self, state):
        """Terminal when car reaches or passes the goal position."""
        return bool(state[0] >= self.goal_pos)

    def get_terminal_states(self):
        """Not meaningful for continuous state space; kept for interface completeness."""
        return None

    def reward_function(self, state, action=None, next_state=None):
        """
        For analysis only: replicate env reward roughly.
        In the actual learning loop we use env's reward directly.
        """
        if next_state is None or action is None:
            # fall back to a simple shaping: 0 at goal, -1 otherwise
            return 0.0 if self.is_terminal(state) else -1.0

        # MountainCarContinuous uses +100 for reaching the goal and
        # a small penalty proportional to squared action each step.
        if self.is_terminal(next_state):
            return 100.0
        return float(-0.1 * (np.array(action, dtype=np.float32) ** 2).sum())

    def is_state_valid(self, state):
        """Check state is within environment observation bounds."""
        pos, vel = state
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        return (low[0] <= pos <= high[0]) and (low[1] <= vel <= high[1])

    def get_next_transitions(self, state, action):
        """
        For model-based algorithms; here transitions are handled by Gymnasium.
        We return a single (next_state, prob=1.0, reward) tuple.
        """
        next_state, reward, terminated, truncated, _ = self.step(action)
        return [(np.array(next_state, dtype=np.float32), 1.0, reward)]
