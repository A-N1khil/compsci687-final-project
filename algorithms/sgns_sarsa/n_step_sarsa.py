import numpy as np
from tqdm.auto import tqdm

from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


class Config:
    def __init__(self, q_init_choice="optimistic", optimistic_value=5.0, epsilon=0.1):
        self.q_init_choice = q_init_choice
        self.optimistic_value = optimistic_value
        self.epsilon = epsilon


class SGNStepSARSA:

    def __init__(
        self,
        env: MDPBase,
        optimal_vf,
        gamma=0.925,
        config=Config(),
        base_config=BaseConfig(42),
    ):
        self.rng = base_config.get_rng()
        self.env = env
        self.gamma = gamma
        self.config = config
        self.optimal_vf = optimal_vf
        self.reset()

    def reset(self):
        self.valid_states = [
            s for s in self.env.get_state_space() if self.env.is_state_valid(s)
        ]
        self.q_sa = {
            s: {a: 0.0 for a in self.env.get_action_space()} for s in self.valid_states
        }
        self.pointers = []
        self.num_steps = 0
        self.num_episodes = 0
        self.mse_data = []

    def optimistic_initialization(self, optimistic_value=5.0):
        for s in self.q_sa:
            for a in self.q_sa[s]:
                self.q_sa[s][a] = optimistic_value

    def random_initialization(self):
        for s in self.q_sa:
            for a in self.q_sa[s]:
                self.q_sa[s][a] = self.rng.uniform(0, 1.0)

    def equal_initialization(self, zeroes=False):
        for s in self.q_sa:
            for a in self.q_sa[s]:
                self.q_sa[s][a] = (
                    0.0 if zeroes else 1.0 / len(self.env.get_action_space())
                )

    def initialize_q(self):
        choice = self.config.q_init_choice
        if choice == "optimistic":
            self.optimistic_initialization(self.config.optimistic_value)
        elif choice == "random":
            self.random_initialization()
        elif choice == "equal":
            self.equal_initialization()
        elif choice == "zeroes":
            self.equal_initialization(zeroes=True)

        # Terminal Q-values = 0
        for terminal_state in self.env.get_terminal_states():
            for a in self.q_sa[terminal_state]:
                self.q_sa[terminal_state][a] = 0.0

    def epsilon_greedy_action(self, state, epsilon):
        actions = self.env.get_action_space()
        best_q = max(self.q_sa[state].values())
        best_actions = [a for a in actions if self.q_sa[state][a] == best_q]

        if self.rng.random() < epsilon:
            return self.rng.choice(actions)  # explore
        return self.rng.choice(best_actions)  # exploit

    def run_episode(self, alpha: float, n_steps: int = 3):
        state = self.rng.choice(self.valid_states)
        action = self.epsilon_greedy_action(state, self.config.epsilon)

        states = [state]
        actions = [action]
        rewards = []

        time = 0
        T = float("inf")
        total_reward = 0

        while True:
            if time < T:
                next_state, reward, done = self.env.step(state, action)
                rewards.append(reward)
                total_reward += reward
                self.num_steps += 1

                if done:
                    T = time + 1
                    states.append(next_state)
                    actions.append(None)
                else:
                    next_action = self.epsilon_greedy_action(
                        next_state, self.config.epsilon
                    )
                    states.append(next_state)
                    actions.append(next_action)
                    state = next_state
                    action = next_action

            tau = time - n_steps + 1

            if tau >= 0:
                # Compute G
                G = 0.0
                for i in range(tau, min(tau + n_steps, T)):
                    G += (self.gamma ** (i - tau)) * rewards[i]

                # Bootstrap if needed
                if tau + n_steps < T:
                    G += (self.gamma**n_steps) * self.q_sa[states[tau + n_steps]][
                        actions[tau + n_steps]
                    ]

                # Update Q
                if tau < T:
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    if a_tau is not None:
                        td_error = G - self.q_sa[s_tau][a_tau]
                        self.q_sa[s_tau][a_tau] += alpha * td_error

            if tau == T - 1:
                break

            time += 1

        self.post_episode()
        return total_reward

    def post_episode(self):
        self.num_episodes += 1
        self.pointers.append(self.num_steps)

        # Compute state-value estimate
        policy = self.get_policy()
        v_estimate = self.compute_value_function(policy)

        # Compute MSE if optimal value function exists
        mse = self.compute_mse(v_estimate, self.optimal_vf)
        self.mse_data.append(mse)

    def run(self, num_episodes: int = 10000, alpha: float = 0.1, n_steps: int = 3):
        self.reset()
        self.initialize_q()

        for _ in tqdm(range(num_episodes), desc="Running SARSA Episodes", leave=False):
            self.run_episode(alpha, n_steps=n_steps)

    def run_n_times(
        self,
        num_times: int = 10,
        num_episodes: int = 10000,
        alpha: float = 0.1,
        n_steps: int = 3,
    ):
        pointer_metadata = []
        mse_metadata = []

        for _ in tqdm(
            range(num_times), desc=f"{num_times} Independent Runs", leave=False
        ):
            self.run(num_episodes, alpha, n_steps=n_steps)
            pointer_metadata.append(self.pointers.copy())
            mse_metadata.append(self.mse_data.copy())

        return np.array(pointer_metadata, dtype=object), np.array(
            mse_metadata, dtype=object
        )

    def get_policy(self):
        policy = {}
        actions = self.env.get_action_space()
        A = len(actions)

        for state, q_values in self.q_sa.items():
            q_list = np.array([q_values[a] for a in actions])
            eps = self.config.epsilon
            max_q = np.max(q_list)
            best_actions = [a for a in actions if q_values[a] == max_q]
            B = len(best_actions)
            prob = {}
            for a in actions:
                if a in best_actions:
                    prob[a] = ((1 - eps) / B) + (eps / A)
                else:
                    prob[a] = eps / A
            policy[state] = prob

        return policy

    def compute_value_function(self, policy):
        """Vπ(s) = Σ_a π(a|s) Q(s,a)."""
        v = {}
        for s in self.valid_states:
            v[s] = sum(
                policy[s][a] * self.q_sa[s][a] for a in self.env.get_action_space()
            )
        return v

    def compute_mse(self, v_estimate, v_true):
        """Compute mean squared error across states."""
        if v_true is None:
            return None
        err = 0.0
        count = 0
        for s in v_estimate:
            err += (v_estimate[s] - v_true[s]) ** 2
            count += 1
        return err / count
