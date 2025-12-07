from algorithms.mcts.MCTSNode import MCTSNode
from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase
from tqdm.notebook import tqdm


class PersistentMCTS:
    def __init__(self, env: MDPBase, gamma=0.925, base_config=BaseConfig()):
        self.env = env
        self.gamma = gamma
        self.nodes = {}  # state -> MCTSNode
        self.rng = base_config.get_rng()

    def get_node(self, state):
        if state not in self.nodes:
            self.nodes[state] = MCTSNode(env=self.env, state=state)
        return self.nodes[state]

    def simulate_from_state(self, root_state, c_value=1.414):
        node = self.get_node(root_state)
        path = [node]

        while node.is_fully_expanded() and node.children:
            parent = node
            node = max(
                parent.children.values(),
                key=lambda child: child.uct_score(parent.visits, c_value),
            )
            path.append(node)

        # --- Expansion ---
        if not node.is_fully_expanded():
            action, child, reward = node.expand()
            path.append(child)
            immediate_reward = reward
        else:
            immediate_reward = 0.0

        # --- Rollout ---
        rollout_reward = self.rollout(path[-1].state)
        total_return = immediate_reward + rollout_reward

        # --- Backpropagation ---
        for i, n in enumerate(reversed(path)):
            n.visits += 1
            n.value += total_return * (self.gamma**i)

        return total_return

    def rollout(self, state):
        total_reward = 0.0
        discount = 1.0

        while not self.env.is_terminal(state):
            action = self.rng.choice(self.env.get_action_space())
            next_state, reward, done = self.env.step(state, action)
            total_reward += discount * reward
            discount *= self.gamma
            state = next_state

        return total_reward

    def run_iterations(self, root_state, num_iterations=500):
        bar = tqdm(
            total=num_iterations, desc=f"MCTS iterations from {root_state}", leave=False
        )
        for _ in range(num_iterations):
            self.simulate_from_state(root_state)
            bar.update(1)
        bar.close()

    def estimate_value(self, state):
        node = self.nodes.get(state)
        if node is None or node.visits == 0:
            return 0.0
        return node.value / node.visits

    def estimate_value_function(self, num_iterations_per_state=500):
        value_function = {}
        valid_states = [
            state
            for state in self.env.get_state_space()
            if self.env.is_state_valid(state)
        ]
        for state in tqdm(valid_states, desc="Estimating value function", leave=False):
            self.run_iterations(state, num_iterations=num_iterations_per_state)
            value_function[state] = self.estimate_value(state)
        return value_function

    def extract_q_policy(self, use_visits=False):
        policy = {}
        actions = self.env.get_action_space()

        for state, node in self.nodes.items():
            best_action = None
            best_q = float("-inf")

            for action in actions:

                if action in node.children:
                    child = node.children[action]

                    if use_visits:
                        q = child.visits
                    else:
                        if child.visits > 0:
                            q = child.value / child.visits
                        else:
                            q = 0.0
                else:
                    q = 0.0  # optimistic or pessimistic

                if q > best_q:
                    best_q = q
                    best_action = action

            policy[state] = best_action

        return policy
