import random

import numpy as np
from tqdm.auto import tqdm

from mdp.cats_v_monsters.cats_vs_monsters import CatsVMonstersMDP


class Config:
    def __init__(self, method='sarsa', q_init_choice='optimistic', optimistic_value=5.0, action_selection='epsilon_greedy',
                 epsilon=0.9):
        self.method = method
        self.q_init_choice = q_init_choice
        self.optimistic_value = optimistic_value
        self.action_selection = action_selection
        self.epsilon = epsilon


class SarsaQL:

    def __init__(self, rng=random.Random(), gamma=0.925, config=Config(), optimal_vf = None):
        self.mdp: CatsVMonstersMDP = CatsVMonstersMDP(rng=rng)
        self.rng = rng
        self.gamma = gamma
        self.q_sa = {state: {action: 0.0 for action in self.mdp.get_action_space()} for state in
                     self.mdp.get_valid_state_space()}
        self.config = config
        self.pointers = []
        self.num_steps = 0
        self.num_episodes = 0
        self.mse_data = []
        self.optimal_vf = optimal_vf

    def reset(self):
        self.q_sa = {state: {action: 0.0 for action in self.mdp.get_action_space()} for state in
                     self.mdp.get_valid_state_space()}
        self.pointers = []
        self.num_steps = 0
        self.num_episodes = 0
        self.mse_data = []

    def optimistic_initialization(self, optimistic_value=5.0):
        for state in self.q_sa:
            for action in self.q_sa[state]:
                self.q_sa[state][action] = optimistic_value

    def random_initialization(self):
        for state in self.q_sa:
            for action in self.q_sa[state]:
                self.q_sa[state][action] = self.rng.uniform(0, 1.0)

    def equal_initialization(self, zeroes=False):
        for state in self.q_sa:
            for action in self.q_sa[state]:
                if zeroes:
                    self.q_sa[state][action] = 0.0
                else:
                    self.q_sa[state][action] = 1.0 / len(self.mdp.get_action_space())

    def initialize_q(self):
        choice = self.config.q_init_choice
        if choice == 'optimistic':
            self.optimistic_initialization(self.config.optimistic_value)
        elif choice == 'random':
            self.random_initialization()
        elif choice == 'equal':
            self.equal_initialization()
        elif choice == 'zeroes':
            self.equal_initialization(zeroes=True)

        # Set terminal states Q-values to 0
        self.q_sa[self.mdp.food] = {action: 0.0 for action in self.mdp.get_action_space()}

    def epsilon_greedy_action(self, state, epsilon=0.9):
        action_space = self.mdp.get_action_space()
        best_q_val = max(self.q_sa[state].values())
        best_actions = [action for action in action_space if self.q_sa[state][action] == best_q_val]
        if self.rng.random() < epsilon:
            # Explore: choose a random action
            return self.rng.choice(action_space)
        else:
            # Exploit: choose the best action
            return self.rng.choice(best_actions)

    def softmax_action(self, state):
        action_space = self.mdp.get_action_space()
        state_q_vals = np.array([self.q_sa[state][action] for action in action_space], dtype=object)

        z = state_q_vals - np.max(state_q_vals)
        exp_q = np.exp(z.astype(np.float64))
        probabilities = exp_q / np.sum(exp_q)

        return self.rng.choices(action_space, weights=probabilities, k=1)[0]

    def select_action(self, state):
        if self.config.action_selection == 'epsilon_greedy':
            return self.epsilon_greedy_action(state, self.config.epsilon)
        elif self.config.action_selection == 'softmax':
            return self.softmax_action(state)
        else:
            raise ValueError(f"Unknown action selection method: {self.config.action_selection}")

    def run_episode(self, alpha: float):
        # Pick initial state
        state = self.rng.choice(self.mdp.get_valid_state_space())
        # Pick initial action
        action = self.select_action(state)
        self.num_steps += 1
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done = self.mdp.take_step(state, action)
            total_reward += reward
            self.num_steps += 1

            if done:
                td_target = reward
                td_error = td_target - self.q_sa[state][action]
                self.q_sa[state][action] += alpha * td_error
                self.post_episode()
                break

            if self.config.method == 'sarsa':
                next_action = self.select_action(next_state)
                td_target = reward + self.gamma * self.q_sa[next_state][next_action]
            elif self.config.method == 'q_learning':
                max_next_q = max(self.q_sa[next_state].values())
                td_target = reward + self.gamma * max_next_q
                next_action = self.select_action(next_state)
            else:
                raise ValueError(f"Unknown method: {self.config.method}")


            td_error = td_target - self.q_sa[state][action]
            self.q_sa[state][action] += alpha * td_error

            state = next_state
            action = next_action

        return total_reward

    def post_episode(self):
        self.pointers.append(self.num_steps)
        # If MSE is to be calculated, then optimal_vf must be provided
        v_estimate = self.compute_value_function(self.get_policy())
        mse = self.compute_mse(v_estimate, self.optimal_vf)
        self.mse_data.append(mse)

    def run(self, num_episodes: int = 10000, alpha: float = 0.1):
        self.initialize_q()
        # Reset
        self.reset()
        for _ in tqdm(range(num_episodes), desc="Running SARSA Episodes", leave=False):
            self.run_episode(alpha)

    def run_20_times(self, num_episodes: int = 10000, alpha: float = 0.1):
        pointer_metadata = []
        mse_metadata = []
        for _ in tqdm(range(20), desc="Running SARSA Episodes", leave=False):
            self.run(num_episodes, alpha)
            pointer_metadata.append(self.pointers.copy())
            mse_metadata.append(self.mse_data.copy())
        return pointer_metadata, mse_metadata

    def get_policy(self):
        policy = {}
        action_space = self.mdp.get_action_space()
        num_actions = len(action_space)

        for state, q_values in self.q_sa.items():
            q_vals_list = np.array([q_values[action] for action in action_space], dtype=object)

            if self.config.action_selection == 'epsilon_greedy':
                epsilon = self.config.epsilon
                max_q = np.max(q_vals_list)
                best_actions = [action for action in action_space if q_values[action] == max_q]
                num_best_actions = len(best_actions)

                action_probabilities = {}
                for action in action_space:
                    if action in best_actions:
                        action_probabilities[action] = ((1 - epsilon) / num_best_actions) + (epsilon / num_actions)
                    else:
                        action_probabilities[action] = epsilon / num_actions
                policy[state] = action_probabilities
            else:
                z = q_vals_list - np.max(q_vals_list)
                exp_q = np.exp(z.astype(np.float64))
                probabilities = exp_q / np.sum(exp_q)
                action_probabilities = {action: prob for action, prob in zip(action_space, probabilities)}
                policy[state] = action_probabilities
        return policy

    def compute_value_function(self, policy):
        value_function = {}
        for state in self.q_sa:
            v = 0.0
            for action, action_prob in policy[state].items():
                v += action_prob * self.q_sa[state][action]
            value_function[state] = v
        return value_function

    def compute_mse(self, v_estimate, v_optimal):
        mse = 0.0
        n = len(v_estimate)
        for state in v_estimate:
            mse += (v_estimate[state] - v_optimal[state]) ** 2
        mse /= n
        return mse

    def get_greedy_policy(self, random_selection=False):
        policy = {}
        action_space = self.mdp.get_action_space()

        for state, q_values in self.q_sa.items():
            max_q = max(q_values.values())
            best_actions = [action for action in action_space if q_values[action] == max_q]
            if random_selection:
                policy[state] = self.rng.choice(best_actions)
            else:
                policy[state] = best_actions[0]

        return policy
