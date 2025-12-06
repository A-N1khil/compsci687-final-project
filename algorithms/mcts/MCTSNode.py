import math


class MCTSNode:
    """Node class for Monte Carlo Tree Search"""

    def __init__(self, env, state, parent=None, action=None):
        # Required attributes
        self.env = env  # The environment (MDP)
        self.state = state  # The state represented by this node
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this node from parent

        # Additional attributes
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.value = 0.0  # Estimated value of this node
        self.untested_actions = self.env.get_action_space(state)  # Actions not yet tried from this node

    def is_fully_expanded(self):
        """Check if all actions have been tried from this node"""
        return len(self.untested_actions) == 0

    def expand(self):
        """Expand the node by trying an untested action"""

        action = self.untested_actions.pop()
        next_state, reward, done = self.env.take_step(self.state, action)

        child = MCTSNode(
            env=self.env,
            state=next_state,
            parent=self,
            action=action
        )

        # Add this child to the list of children
        self.children.append(child)
        return child

    def best_child(self, c_value=1.414):
        """Select the best child node based on UCT value"""

        best_child = None
        best_score = float('-inf')

        for child in self.children:
            exploitation_score = child.value / (child.visits + 1e-8)  # Numerical stability
            exploration_score = c_value * math.sqrt(
                math.log(self.visits + 1) / child.visits + 1e-8)  # Numerical stability
            child_score = exploitation_score + exploration_score

            if child_score > best_score:
                best_score = child_score
                best_child = child

        return best_child

    def update(self, reward):
        """Update the node's value and visit count based on the received reward"""
        self.visits += 1
        self.value += reward