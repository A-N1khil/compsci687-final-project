import numpy as np
import gymnasium as gym
from abc import ABC
from mdp.mdp_base import MDPBase


class CartPoleMDP(MDPBase, ABC):

    # initialising CartPole-v1 MDP environment
    def __init__(self, render_mode=None, seed=42):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env.reset(seed=seed)

        # storing action and state dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_list = list(range(self.env.action_space.n))

        # setting runtime state variables
        self.current_state = None
        self.last_info = {}

    # returning continuous state space description
    def get_state_space(self):
        return f"Continuous, dim={self.state_dim}"

    # returning list of discrete actions
    def get_action_space(self):
        return self.action_list

    # stepping environment using given action index
    def step(self, state, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.current_state = obs
        self.last_info = info

        return obs, reward, done

    # resetting environment and returning initial state
    def reset(self):
        obs, _ = self.env.reset()
        self.current_state = obs
        return obs

    # returning terminal state list (handled internally by env)
    def get_terminal_states(self):
        return []

    # checking if a state is terminal (delegating to env termination logic)
    def is_terminal(self, state):
        return False

    # returning reward for a step (CartPole gives +1 per step)
    def reward_function(self, state, action=None, next_state=None):
        return 1.0

    # checking validity of a state vector
    def is_state_valid(self, state):
        return (
            isinstance(state, (np.ndarray, list, tuple)) and
            len(state) == self.state_dim and
            np.all(np.isfinite(state))
        )

    # returning empty transitions list (model-free environment)
    def get_next_transitions(self, state, action):
        return []
