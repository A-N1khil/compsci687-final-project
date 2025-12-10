import numpy as np
from tqdm.auto import tqdm

from mdp.base_config import BaseConfig


class MCConfig:
    def __init__(
        self, epsilon=0.1, weight_init_choice="optimistic", optimistic_value=5.0
    ):
        self.epsilon = epsilon
        self.weight_init_choice = weight_init_choice
        self.optimistic_value = optimistic_value


class NStepSarsaMC:
<<<<<<< HEAD
    def __init__(
=======
    def __init__(  # noqa: PLR0913 - constructor needs full hyperparameter set
>>>>>>> 851d10d (Merge conflicts)
        self,
        env,
        config: MCConfig,
        feature_fn,
        num_features,
        gamma=1.0,
        base_config=BaseConfig(seed=42),
    ):
        self.rng = base_config.get_numpy_rng()
        self.env = env
        self.config = config
        self.feature_fn = feature_fn
        self.num_features = num_features
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.actions = list(range(self.env.action_space.n))
        self.num_actions = len(self.actions)
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.w = np.zeros((self.num_actions, self.num_features), dtype=np.float64)
        self.pointers = []
        self.num_steps = 0
        self.num_episodes = 0
        self.episode_rewards = []

        # Weight initialization
        if self.config.weight_init_choice == "optimistic":
            self.w.fill(self.config.optimistic_value)
        elif self.config.weight_init_choice == "zero":
            self.w.fill(0.0)
        elif self.config.weight_init_choice == "random":
            self.w = self.rng.uniform(0, 1.0, size=self.w.shape)

    def q_values(self, state):
        phi = self.feature_fn(state)
        return self.w @ phi  # shape: (num_actions,)

    def q_value(self, state, action):
        idx = self.action_to_index[action]
        phi = self.feature_fn(state)
        return float(self.w[idx] @ phi)

    def epsilon_greedy_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(self.actions)
        q = self.q_values(state)
        best_idxs = np.flatnonzero(q == np.max(q))
        return self.actions[int(self.rng.choice(best_idxs))]

    def run_episode(self, alpha: float, n_steps: int = 3, max_steps: int = 10000):
        state = self.env.reset()
        action = self.epsilon_greedy_action(state, self.config.epsilon)

        states = [state]
        actions = [action]
        rewards = []

        time = 0
        T = 1e9  # Infinity
        total_reward = 0.0

        while True:
            if time < T:
                next_state, reward, done, _ = self.env.step(state=None, action=action)
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
                G = 0.0
                upper = min(tau + n_steps, int(T))
                for i in range(tau, upper):
                    G += (self.gamma ** (i - tau)) * rewards[i]

                if tau + n_steps < T:
                    s_boot = states[tau + n_steps]
                    a_boot = actions[tau + n_steps]
                    G += (self.gamma**n_steps) * self.q_value(s_boot, a_boot)

                s_tau = states[tau]
                a_tau = actions[tau]
                if a_tau is not None:
                    phi_tau = self.feature_fn(s_tau)
                    idx = self.action_to_index[a_tau]
                    td_error = G - (self.w[idx] @ phi_tau)
                    self.w[idx] += alpha * td_error * phi_tau

            if tau == T - 1 or time > max_steps:
                break

            time += 1

        self.num_episodes += 1
        self.pointers.append(self.num_steps)
        self.episode_rewards.append(total_reward)
        return total_reward

    def run(self, num_episodes=500, alpha=0.1, n_steps=3):
        self.reset()
        for _ in tqdm(range(num_episodes), desc="Running SARSA Episodes"):
            self.run_episode(alpha, n_steps=n_steps)


# ---------- Tile Coding Helper ----------
class MountainCarSARSAHelpers:
    def __init__(self, env, num_tilings=8, num_tiles=8):
        self.env = env
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        obs_space = env.env.observation_space
        self.lows = obs_space.low
        self.highs = obs_space.high
        self.n_dims = len(self.lows)
        self.num_features = num_tilings * (num_tiles**self.n_dims)
        self.tile_width = (self.highs - self.lows) / (num_tiles - 1)

    def get_active_tiles(self, state):
        active_tiles = []
        for tiling in range(self.num_tilings):
            coords = []
            for i in range(self.n_dims):
                offset = (tiling / self.num_tilings) * self.tile_width[i]
                coord = int(
                    np.floor((state[i] + offset - self.lows[i]) / self.tile_width[i])
                )
                coords.append(coord)
            # map multi-dim coords to single index
            idx = (
                coords[0]
                + coords[1] * self.num_tiles
                + tiling * (self.num_tiles**self.n_dims)
            )
            active_tiles.append(idx)
        return np.array(active_tiles, dtype=int)

    def get_tile_coding_params(self):
        def feature_fn(state):
            phi = np.zeros(self.num_features, dtype=np.float64)
            phi[self.get_active_tiles(state)] = 1.0
            return phi

        return feature_fn, self.num_features
