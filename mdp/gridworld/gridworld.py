from mdp.base_config import BaseConfig


class GridWorldMDP:
    """This class defines the MDP for the GridWorld environment."""

    def __init__(
        self,
        rows = 5,
        cols = 5,
        start = (0, 0),
        goal = (4, 4),
        overrides = None,
    ):
        """Initializes the GridWorld MDP with optional override settings."""
        base_config = BaseConfig(seed=42)
        self.rows = rows
        self.cols = cols
        self.start = start
        self.state_space = [(r, c) for r in range(rows) for c in range(cols)]
        self.action_space = ["up", "down", "left", "right"]

        # Terminal state
        self.goal = goal

        # Forbidden cells
        self.furniture = [(2, 2), (3, 2)]

        # Water state
        self.water = [(4, 2)]

        # Discount factor
        self.gamma = 0.9


        # Rewards
        self.goal_reward = 10.0
        self.water_reward = -10.0
        self.step_cost = 0.0

        # Intended actions
        self.intended_actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        # Veering actions
        self.veering_actions_left = {
            "up": (0, -1),     # up veers left to left
            "down": (0, 1),    # down veers left to right
            "left": (1, 0),    # left veers left to down
            "right": (-1, 0),  # right veers left to up
        }

        self.veering_actions_right = {
            "up": (0, 1),     # up veers right to right
            "down": (0, -1),  # down veers right to left
            "left": (-1, 0),  # left veers right to up
            "right": (1, 0),  # right veers right to down
        }

        # Transition model
        self.direction_probabilities = {
            "intended": 0.8,
            "left": 0.05,
            "right": 0.05,
            "stay": 0.10,
        }

        self.rng = base_config.get_rng()

    def get_state_space(self):
        """Returns state space"""
        return self.state_space

    def get_valid_state_space(self):
        """Returns valid state space"""
        return [state for state in self.state_space if self.is_valid_state(state)]

    def get_action_space(self):
        """Returns action space"""
        return self.action_space

    def reset(self):
        """Reset the environment to the start state"""
        return self.start

    def reward_fn(self, state):
        """Reward function for the GridWorld MDP"""
        if state == self.goal:
            return self.goal_reward
        if state in self.water:
            return self.water_reward
        return self.step_cost

    def is_terminal_state(self, state):
        """A state is terminal if it is the goal state"""
        return state == self.goal

    def is_valid_state(self, state):
        """A state is valid if it is within the bounds and not a forbidden cell"""
        r, c = state
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return state not in self.furniture

    def _apply_action(self, state, direction_coords):
        """Apply direction to a state and return the new state"""
        r, c = state
        dr, dc = direction_coords
        candidate = (r + dr, c + dc)
        if not self.is_valid_state(candidate):
            return state
        return candidate

    def get_next_transitions(self, state, action):
        """Given a state and action, return possible next states and their probabilities."""
        if self.is_terminal_state(state):
            return [(state, 1.0, self.reward_fn(state))]

        next_transitions = []
        for direction, prob in self.direction_probabilities.items():
            if direction == "intended":
                direction_coords = self.intended_actions[action]
            elif direction == "left":
                direction_coords = self.veering_actions_left[action]
            elif direction == "right":
                direction_coords = self.veering_actions_right[action]
            else:
                direction_coords = (0, 0)

            next_state = self._apply_action(state, direction_coords)
            reward = self.reward_fn(next_state)
            next_transitions.append((next_state, prob, reward))

        return next_transitions

    def sample_transition(self, state, action):
        """Sample a transition tuple for a given state/action."""
        transitions = self.get_next_transitions(state, action)
        next_states,probs,rewards = [],[],[]
        for ns, p, r in transitions:
            next_states.append(ns)
            probs.append(p)
            rewards.append(r)
        i = self.rng.choices(range(len(next_states)), weights=probs, k=1)[0]
        next_state,reward = next_states[i],rewards[i]
        terminal = self.is_terminal_state(next_state)
        return next_state, reward, terminal

