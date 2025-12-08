import numpy as np
from tqdm.auto import tqdm

from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


class Config:
    def __init__(self, q_init_choice='optimistic', optimistic_value=5.0, action_selection='epsilon_greedy',
                 epsilon=0.9):
        self.q_init_choice = q_init_choice
        self.optimistic_value = optimistic_value
        self.action_selection = action_selection
        self.epsilon = epsilon


class SGNStepSARSA:

    def __init__(self, env: MDPBase, gamma=0.925, config=Config(), optimal_vf=None, base_config=BaseConfig(42)):
        self.rng = base_config.get_rng()
        self.env = env
        self.gamma = gamma
        self.config = config
        self.optimal_vf = optimal_vf
        self.reset()

    def reset(self):
        self.valid_states = [state for state in self.env.get_state_space() if self.env.is_state_valid(state)]
        self.q_sa = {state: {action: 0.0 for action in self.env.get_action_space()} for state in self.valid_states}
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
                    self.q_sa[state][action] = 1.0 / len(self.env.get_action_space())

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
        for terminal_state in self.env.get_terminal_states():
            for action in self.q_sa[terminal_state]:
                self.q_sa[terminal_state][action] = 0.0

    def epsilon_greedy_action(self, state, epsilon=0.9):
        action_space = self.env.get_action_space()
        best_q_val = max(self.q_sa[state].values())
        best_actions = [action for action in action_space if self.q_sa[state][action] == best_q_val]
        if self.rng.random() < epsilon:
            # Explore: choose a random action
            return self.rng.choice(action_space)
        else:
            # Exploit: choose the best action
            return self.rng.choice(best_actions)

    def softmax_action(self, state):
        action_space = self.env.get_action_space()
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

    def run_episode(self, alpha: float, n_steps: int = 3):
        # Pick initial state
        state = self.rng.choice(self.valid_states)
        # Pick initial action
        action = self.select_action(state)

        # Initialize n-step buffer
        states = [state]
        actions = [action]
        rewards = [0.0] # reward at time t=0 is 0
        time = 0
        T = float('inf') # time when episode ends

        self.num_steps += 1
        total_reward = 0

        while True:
            if time < T: # until episode ends
                next_state, reward, done = self.env.step(state, action)
                rewards.append(reward)
                total_reward += reward
                self.num_steps += 1

                if done:
                    T = time + 1
                else:
                    next_action = self.select_action(next_state)
                    states.append(next_state)
                    actions.append(next_action)

            tau = time - n_steps + 1
            if tau >= 0:
                G = 0.0
                # Calculate G as the sum of rewards
                for i in range(tau + 1, min(tau + n_steps, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]

                # If episode not ended, add the estimated value of the next state-action pair
                if tau + n_steps < T:
                    next_time_step = tau + n_steps
                    G += (self.gamma ** n_steps) * self.q_sa[states[next_time_step]][actions[next_time_step]]

                # Perform Q-value update
                tau_s = states[tau]
                tau_a = actions[tau]
                td_error = G - self.q_sa[tau_s][tau_a]
                self.q_sa[tau_s][tau_a] += alpha * td_error

            if tau == T - 1:
                break # all updates done

        # Note to self: Do not call after done inside the loop, as it will mess up the pointers
        # Since the loop runs until all updates are done
        self.post_episode()
        return total_reward

    def post_episode(self):
        self.num_episodes += 1
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
        action_space = self.env.get_action_space()
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
        action_space = self.env.get_action_space()

        for state, q_values in self.q_sa.items():
            max_q = max(q_values.values())
            best_actions = [action for action in action_space if q_values[action] == max_q]
            if random_selection:
                policy[state] = self.rng.choice(best_actions)
            else:
                policy[state] = best_actions[0]

        return policy
