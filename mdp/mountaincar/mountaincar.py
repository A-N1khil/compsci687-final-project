import gymnasium as gym # Using Gymnasium instead of Gym since Gym is deprecated and caused numpy warnings
import numpy as np
from mdp.mdp_base import MDPBase


class MountainCarMDP(MDPBase):
    """
    This class defines the MDP for the MountainCar-v0 environment using Gymnasium,
    which is the actively maintained successor to OpenAI Gym.
    """

    def __init__(self, seed=None):
        # Gymnasium handles continuous physics internally
        self.env = gym.make("MountainCar-v0")

        # Shape: (position, velocity)
        self.state_space = self.env.observation_space

        # Actions: 0=push left, 1=no push, 2=push right
        self.action_space = self.env.action_space

        # Goal position
        self.goal_pos = 0.5

        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        """Reset the environment and return the initial state."""
        state, _ = self.env.reset()
        return np.array(state, dtype=np.float32)

    def get_state_space(self):
        """Return the continuous state space."""
        return self.state_space

    def get_action_space(self):
        """Return the discrete action space."""
        return self.action_space

<<<<<<< HEAD
    # def step(self, action):
    #     """
    #     Take a step in the environment. 
    #     """
    #     next_state, reward, terminated, truncated, info = self.env.step(action)
    #     done = terminated or truncated # Gymnasium has seperate flags for success and timeout , but we need to count both as done

    #     return (np.array(next_state, dtype=np.float32),reward,done,info,)
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Aggressive reward shaping
        position, velocity = next_state
        shaped_reward = reward + 100.0 * (position - self.env.observation_space.low[0])

        # Encourage reaching the goal early
        if self.is_terminal(next_state):
            shaped_reward += 500.0

        return np.array(next_state, dtype=np.float32), shaped_reward, done, info

<<<<<<< HEAD
=======
    def step(self, state, action):
        """
        Take a step in the environment. 
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated # Gymnasium has separate flags for success and timeout , but we need to count both as done

        return (np.array(next_state, dtype=np.float32),reward,done,info,)
>>>>>>> develop

=======
>>>>>>> 851d10d (Merge conflicts)
    def is_terminal(self, state):
        """Terminal when car reaches or passes the goal."""
        return bool(state[0] >= self.goal_pos)

    def get_terminal_states(self):
        """Not applicable for continuous domains."""
        return None

    def reward_function(self, state, action=None, next_state=None):
        """-1 per step until reaching the goal, same as MountainCar default."""
        return 0.0 if self.is_terminal(state) else -1.0

    def is_state_valid(self, state):
        """
        Since MountainCar is continuous, any state within the environment bounds is valid.
        """
        pos, vel = state
        return (self.env.observation_space.low[0] <= pos <= self.env.observation_space.high[0] and
                self.env.observation_space.low[1] <= vel <= self.env.observation_space.high[1])

    def get_next_transitions(self, state, action):
        """
        Transition dynamics are handled internally by Gymnasium,so each step is returned as a single outcome with prob 1.0.

        Note: This method is not used by Actor-Critic as it is model free, 
        but this method is needed to satisfy the MDPBase interface used for the project. 
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return [(np.array(next_state, dtype=np.float32), 1.0, reward)]
