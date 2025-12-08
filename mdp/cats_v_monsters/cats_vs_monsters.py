from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


class CatsVMonstersMDP(MDPBase):
    """This class defines the MDP for the Cats vs Monsters environment."""

    def __init__(self, rows=5, cols=5, base_config=BaseConfig(seed=42), gamma=0.9):
        """We define the MDP for the Cats vs Monsters environment."""
        self.rows = rows
        self.cols = cols
        self.state_space = [(r, c) for r in range(rows) for c in range(cols)]
        self.action_space = ["up", "down", "left", "right"]

        # Terminal state
        self.food = (4, 4)

        # Location for monsters
        self.monsters = [(0, 3), (4, 1)]

        # Forbidden cells
        self.furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]

        # Now we define the intended actions
        self.intended_actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        self.veering_actions_left = {
            "up": (0, -1),  # up veers left to left
            "down": (0, 1),  # down veers left to right
            "left": (1, 0),  # left veers left to down
            "right": (-1, 0),  # right veers left to up
        }

        self.veering_actions_right = {
            "up": (0, 1),  # up veers right to right
            "down": (0, -1),  # down veers right to left
            "left": (-1, 0),  # left veers right to up
            "right": (1, 0),  # right veers right to down
        }

        self.move_probabilities = {
            "intended": 0.7,
            "left": 0.12,
            "right": 0.12,
            "stay": 0.06,
        }

        self.rng = base_config.get_rng()
        self.gamma = gamma

    def get_gamma(self):
        return self.gamma

    # noinspection PyUnusedLocal
    def reward_function(self, state, action=None, next_state=None):
        if state == self.food:
            return 10  # Reward for reaching food
        elif state in self.monsters:
            return -8  # Penalty for encountering a monster
        else:
            return -0.05  # Small penalty for each move

    def is_terminal(self, state):
        """A state is terminal if it is the food location"""
        return state == self.food

    def is_valid_state(self, state):
        """Check if the state is within bounds and not a forbidden cell"""
        r, c = state
        if (
                (0 <= r < self.rows)
                and (0 <= c < self.cols)
                and (state not in self.furniture)
        ):
            return True
        return False

    def get_valid_state_space(self):
        """Return only the valid states in the state space"""
        return [state for state in self.state_space if self.is_valid_state(state)]

    def get_next_transitions(self, state, action):
        """Given a state and action, return possible next states and their probabilities."""
        if self.is_terminal(state):
            return [(state, 1.0, 0)]  # No transitions from terminal state

        r, c = state

        outcomes = []
        for move_type, prob in self.move_probabilities.items():
            if move_type == "intended":
                dr, dc = self.intended_actions[action]
            elif move_type == "left":
                dr, dc = self.veering_actions_left[action]
            elif move_type == "right":
                dr, dc = self.veering_actions_right[action]
            else:  # stay
                dr, dc = (0, 0)

            new_state = (r + dr, c + dc)
            if not self.is_valid_state(new_state):
                new_state = state  # Stay in the same state if invalid

            # Assign the reward here for quick calculation later
            reward = self.reward_function(new_state)

            outcomes.append((new_state, prob, reward))

        return outcomes

    # Edit 1: For Monte Carlo simulations in HW4, adding a method to get all possible actions
    def get_action_space(self):
        return self.action_space

    # Edit 3: For Monte Carlo simulations in HW4, adding a method to take a step
    def take_step(self, state, action):
        # First get all the possible transitions from a current state
        transitions = self.get_next_transitions(state, action)

        # Unzip the transitions into next states, probabilities, and rewards
        next_states, probs, rewards = zip(*transitions)

        # Choose the next state based on the probabilities
        next_state_index = self.rng.choices(range(len(next_states)), weights=probs, k=1)[0]

        # Select the next state and the corresponding reward
        next_state = next_states[next_state_index]
        reward = rewards[next_state_index]

        # Check if the next state is terminal
        done = self.is_terminal(next_state)

        # Return the next state, reward, and done flag
        return next_state, reward, done

    def get_state_space(self):
        return self.state_space

    def step(self, state, action):
        return self.take_step(state, action)

    def get_terminal_states(self):
        return [self.food]

    def is_state_valid(self, state):
        return self.is_valid_state(state)
