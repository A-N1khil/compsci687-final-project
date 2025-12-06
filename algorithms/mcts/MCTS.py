from algorithms.mcts.MCTSNode import MCTSNode
from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase
from tqdm.notebook import tqdm


class MCTS:

    def __init__(self, env: MDPBase, base_config=BaseConfig()):
        self.env = env
        self.rng = base_config.get_rng()

    def search(self, root_state, num_simulations=500, c_value=1.414):

        root = MCTSNode(env=self.env, state=root_state)
        for _ in range(num_simulations):
            node = root

            # Perform selection to find a node to expand, i.e a node that has actions that have not been tried yet
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c_value)

            # Expand the node if it is not fully expanded
            if not node.is_fully_expanded():
                node = node.expand()

            # Simulate a random playout from the expanded node
            reward = node.incoming_reward + self.rollout(node.state)

            # Backpropagate the reward up the tree
            node.backpropagate(reward)

        return root, self.best_action(root)

    def best_action(self, root: MCTSNode):
        best_action = None
        max_visits = -1
        for child in root.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = child.action
        return best_action

    def rollout(self, state):
        current = state
        total_reward = 0
        done = False

        while not done:
            actions = (
                self.env.get_action_space()
            )  # For simplicity, we assume all actions are available
            action = self.rng.choice(actions)
            next_state, reward, done = self.env.step(current, action)

            total_reward += reward
            current = next_state

        return total_reward

    def estimate_value_function(
        self, num_tries=1000, num_simulations=500, c_value=1.414
    ):

        state_space = self.env.get_state_space()
        value_function = {state: 0.0 for state in state_space}
        valid_states = [
            state for state in state_space if self.env.is_state_valid(state)
        ]

        total_reward = {state: 0.0 for state in valid_states}
        total_visits = {state: 0 for state in valid_states}

        bar = tqdm(total=num_tries, desc="Estimating value function", leave=False)
        for _ in range(num_tries):
            root_state = self.rng.choice(valid_states)
            root, _ = self.search(
                root_state, num_simulations=num_simulations, c_value=c_value
            )
            # Update total rewards and visits for the root state
            total_reward[root_state] += root.value
            total_visits[root_state] += root.visits
            # Update progress bar
            bar.update(1)

        # Close progress bar
        bar.close()
        bar.disable = True

        for state in valid_states:
            if total_visits[state] > 0:
                value_function[state] = total_reward[state] / total_visits[state]
            else:
                value_function[state] = 0.0  # or some default value

        return value_function
