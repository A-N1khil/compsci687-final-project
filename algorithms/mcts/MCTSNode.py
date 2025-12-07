import math

from mdp.mdp_base import MDPBase


class MCTSNode:
    """Node class for Monte Carlo Tree Search"""

    def __init__(self, env: MDPBase, state, parent=None):
        # Required attributes
        self.env = env  # The environment (MDP)
        self.state = state  # The state represented by this node
        self.parent = parent  # Parent node

        # Additional attributes
        self.children = {}  # Dictionary of child nodes: action -> MCTSNode
        self.visits = 0  # Number of times this node has been visited
        self.value = 0.0  # Estimated value of this node
        self.untested_actions = self.env.get_action_space().copy()  # Actions not yet tried from this node

        self.incoming_reward = 0.0  # Reward received upon reaching this node

    def is_fully_expanded(self):
        """Check if all actions have been tried from this node"""
        return len(self.untested_actions) == 0

    def expand(self):
        """Expand the node by trying an untested action"""

        if not self.untested_actions:
            raise Exception("No untested actions available to expand.")

        action = self.untested_actions.pop()
        next_state, reward, done = self.env.step(self.state, action)

        child = MCTSNode(
            env=self.env,
            state=next_state,
            parent=self,
        )

        # Store the incoming reward for this child node
        child.incoming_reward = reward

        # Add this child to the list of children
        self.children[action] = child
        return action, child, reward

    def best_child(self, c_value=1.414):
        """Select the best child node based on UCT value"""

        best_child = None
        best_score = float('-inf')

        for child in self.children.values():
            score = child.uct_score(self.visits, c_value)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def uct_score(self, parent_visits, c_value=1.414):
        """Calculate the UCT score for this node"""
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes

        exploitation = self.value / self.visits
        exploration = c_value * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def backpropagate(self, reward):
        """Update the node's value and visit count based on the received reward"""
        self.visits += 1
        self.value += reward

        # Propagate the reward to the parent node
        if self.parent:
            self.parent.backpropagate(reward)