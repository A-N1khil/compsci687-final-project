from openpyxl.styles.builtins import total
from tqdm.notebook import tqdm

from algorithms.mcts.MCTSNode import MCTSNode
from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


class MCTS:

    def __init__(self, env: MDPBase, base_config=BaseConfig(), rollout_steps=50):
        self.env = env
        self.rng = base_config.get_rng()
        self.rollout_steps = rollout_steps

    def search(self, root_state, num_simulations=500, c_value=1.414):

        root = MCTSNode(env=self.env, state=root_state)
        for _ in range(num_simulations):
            node = root

            # Perform selection to find a node to expand, i.e a node that has actions that have not been tried yet
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c_value)

            # Expand the node if it is not fully expanded
            incoming_reward = 0.0
            if not node.is_fully_expanded():
                action, child, incoming_reward = node.expand()
                rollout_reward = self.rollout(child.state)
                total_reward = incoming_reward + rollout_reward
                child.backpropagate(total_reward)
            else:
                node.backpropagate(self.rollout(node.state))

        return root, self.best_action(root)

    def best_action(self, root: MCTSNode):
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        return best_action

    def rollout(self, state):
        current = state
        total_reward = 0
        time = 0
        done = False

        while not done and time < self.rollout_steps:
            actions = self.env.get_action_space()  # For simplicity, we assume all actions are available
            action = self.rng.choice(actions)
            next_state, reward, done = self.env.step(current, action)

            total_reward += (self.env.get_gamma() ** time) * reward
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

    def generate_value_function_and_policy(self, num_simulations=500, c_value=1.414):

        # Get all valid states
        valid_states = [
            s for s in self.env.get_state_space() if self.env.is_state_valid(s)
        ]

        # Shuffle so we sample uniformly without replacement
        self.rng.shuffle(valid_states)

        value_function = {}
        policy = {}

        bar = tqdm(valid_states, desc="Evaluating all states", leave=False)

        for state in bar:

            # Run MCTS rooted at this state
            root, _ = self.search(
                state, num_simulations=num_simulations, c_value=c_value
            )

            # ----- Extract value estimate V(s) -----
            V_s = root.value / max(root.visits, 1)
            value_function[state] = V_s

            best_a = None
            best_q = -float('inf')
            for action, child in root.children.items():
                if child.visits > 0:
                    avg_value = child.value / child.visits
                else:
                    avg_value = -float('inf')  # ignore unvisited actions

                if avg_value > best_q:
                    best_q = avg_value
                    best_a = action

            policy[state] = best_a

        bar.close()
        bar.disable = True

        return value_function, policy

    def mcts_diagnostics(self, root_state, num_simulations=500, c_value=1.414):
        """
        Run MCTS on a single root_state and collect diagnostic information:
        - Root value convergence
        - Q-value estimates for each action
        - Policy stability (best action over time)
        """

        root = MCTSNode(env=self.env, state=root_state)  # create root node
        actions = self.env.get_action_space()

        # Storage for diagnostics
        root_value_history = []
        q_value_history = {a: [] for a in actions}
        policy_history = []

        for _ in tqdm(range(num_simulations), desc="MCTS Diagnostics"):
            node = root

            # Find the node to expand
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c_value)

            # Expand
            if not node.is_fully_expanded():
                action, child, reward = node.expand()
                node_to_backprop = child
            else:
                reward = 0.0
                node_to_backprop = node

            # Rollout
            rollout_reward = self.rollout(node_to_backprop.state)
            total_reward = reward + rollout_reward

            # Backpropagation
            node_to_backprop.backpropagate(total_reward)

            # Add root value history
            root_value_history.append(root.value / max(1, root.visits))

            # 2. Q-value estimates
            for action in actions:
                if action in root.children:
                    child = root.children[action]
                    q_value = child.value / max(1, child.visits)
                else:
                    q_value = 0.0
                q_value_history[action].append(q_value)

            # 3. Policy stability (greedy action)
            if root.children:
                best_action = max(
                    root.children.items(),
                    key=lambda ac: ac[1].value / max(1, ac[1].visits)
                )[0]
            else:
                best_action = None
            policy_history.append(best_action)

        return root_value_history, q_value_history, policy_history

    def mcts_diagnostics_multiple(self, root_states, num_simulations=500, c_value=1.414):
        diagnostics = {}
        for state in tqdm(root_states, desc="MCTS Diagnostics Multiple States"):
            diagnostics[state] = self.mcts_diagnostics(state, num_simulations, c_value)
        return diagnostics
