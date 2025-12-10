import numpy as np
from tqdm.auto import tqdm
from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


class LinearQFn:
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.theta = np.zeros((num_actions, state_dim), dtype=np.float64)

    def __call__(self, state, action):
        return np.dot(self.theta[action], state)

    def gradient(self, state, action):
        grad = np.zeros_like(self.theta)
        grad[action] = state
        return grad


class NStepSarsaCP:

    def __init__(self, env: MDPBase, gamma=0.99, epsilon=1, base_config=BaseConfig(42)):
        self.rng = base_config.get_rng()
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_dim = env.state_dim
        self.n_actions = len(env.get_action_space())
        self.q_func = LinearQFn(self.state_dim, self.n_actions)

        self.reset()

    def reset(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.episode_rewards = []
        self.discounted_rewards = []
        self.q_func = LinearQFn(self.state_dim, self.n_actions)

    def normalize_state(self, state):
        # CartPole normalization and clipping
        s = np.array(state, dtype=np.float32)
        s[0] /= 2.4  # cart position
        s[1] /= 2.0  # cart velocity (approx)
        s[2] /= 0.2095  # pole angle ~ 12°
        s[3] /= 2.0  # pole angular velocity (approx)
        return np.clip(s, -1, 1)

    def epsilon_greedy_action(self, features, epsilon):
        q_values = [self.q_func(features, a) for a in range(self.n_actions)]
        if any(not np.isfinite(q) for q in q_values):
            return self.rng.choice(range(self.n_actions))
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        if not best_actions:
            return self.rng.choice(range(self.n_actions))
        if self.rng.random() < epsilon:
            return self.rng.choice(range(self.n_actions))
        return self.rng.choice(best_actions)

    def run_episode(self, alpha: float, n_steps: int = 3):
        state_features = self.normalize_state(self.env.reset())
        epsilon = max(0.01, self.epsilon * (0.995**self.num_episodes))
        action = self.epsilon_greedy_action(state_features, epsilon)

        states = [state_features]
        actions = [action]
        rewards = []

        time = 0
        T = 1e11  # Infinity
        total_reward = 0

        while True:
            if time < T:
                next_state_features, reward, done = self.env.step(
                    states[time], actions[time]
                )
                next_state_features = self.normalize_state(next_state_features)
                rewards.append(reward)
                total_reward += reward
                self.num_steps += 1

                if done:
                    T = time + 1
                    states.append(next_state_features)
                    actions.append(None)
                else:
                    next_action = self.epsilon_greedy_action(
                        next_state_features, epsilon
                    )
                    states.append(next_state_features)
                    actions.append(next_action)
                    state_features = next_state_features
                    action = next_action

            tau = time - n_steps + 1
            if tau >= 0:
                # Compute n-step return
                G = sum(
                    (self.gamma ** (i - tau)) * rewards[i]
                    for i in range(tau, min(tau + n_steps, T))
                )

                # Bootstrap if not terminal
                if tau + n_steps < T and actions[tau + n_steps] is not None:
                    G += (self.gamma**n_steps) * self.q_func(
                        states[tau + n_steps], actions[tau + n_steps]
                    )

                # Semi-gradient update
                s_tau = states[tau]
                a_tau = actions[tau]
                if a_tau is not None:
                    td_error = G - self.q_func(s_tau, a_tau)
                    self.q_func.theta[a_tau] += alpha * s_tau * td_error

            if tau == T - 1:
                break
            time += 1

        # Track rewards
        self.episode_rewards.append(total_reward)
        discounted_reward = sum(r * (self.gamma**t) for t, r in enumerate(rewards))
        self.discounted_rewards.append(discounted_reward)
        self.num_episodes += 1

        return total_reward

    def run(self, num_episodes: int = 1000, alpha: float = 0.01, n_steps: int = 3):
        self.reset()
        for _ in tqdm(range(num_episodes), desc="Running SARSA Episodes", leave=False):
            self.run_episode(alpha, n_steps=n_steps)
