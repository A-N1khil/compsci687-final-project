from abc import ABC, abstractmethod


class MDPBase(ABC):
    """Base class for Markov Decision Processes (MDPs)."""

    @abstractmethod
    def get_state_space(self):
        """Return the state space of the MDP."""
        raise NotImplementedError("The get_state_space method must be implemented by the subclass.")

    def get_action_space(self):
        """Return the action space of the MDP."""
        raise NotImplementedError("The get_action_space method must be implemented by the subclass.")

    @abstractmethod
    def step(self, state, action):
        """Take an action in the MDP and return the next state, reward, done, and info."""
        raise NotImplementedError("The step method must be implemented by the subclass.")

    @abstractmethod
    def get_terminal_states(self):
        """Return the terminal states of the MDP."""
        raise NotImplementedError("The get_terminal_states method must be implemented by the subclass.")

    @abstractmethod
    def is_terminal(self, state):
        """Check if a given state is terminal."""
        raise NotImplementedError("The is_terminal method must be implemented by the subclass.")

    @abstractmethod
    def reward_function(self, state, action=None, next_state=None):
        """Return the reward for a given state, action, and next state."""
        raise NotImplementedError("The reward_function method must be implemented by the subclass.")