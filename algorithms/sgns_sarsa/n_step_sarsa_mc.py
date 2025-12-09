import numpy as np
from tqdm.auto import tqdm

from mdp.base_config import BaseConfig
from mdp.mdp_base import MDPBase


# noinspection DuplicatedCode
class MCConfig:
    def __init__(
        self, epsilon=0.1, weight_init_choice="optimistic", optimistic_value=5.0
    ):
        self.epsilon = epsilon
        self.weight_init_choice = weight_init_choice
        self.optimistic_value = optimistic_value


# noinspection DuplicatedCode
class NStepSarsaMC:

    def __init__(
        self,
        env: MDPBase,
        config: MCConfig,
        feature_fn,
        num_features,
        gamma=0.925,
        base_config=BaseConfig(42),
    ):
        self.rng = base_config.get_numpy_rng()
        self.env = env
        self.config = config
        self.feature_fn = feature_fn
        self.num_features = num_features
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.actions = list(range(self.env.get_action_space().n))
        self.num_actions = len(self.actions)
        # Map actions to indices for easier handling
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}

        # weight initialization
        self.w = np.zeros((self.num_actions, self.num_features), dtype=np.float64)

        self.pointers = []
        self.num_steps = 0
        self.num_episodes = 0
        self.episode_rewards = []

    def optimistic_initialization(self, optimistic_value=1.0):
        self.w.fill(optimistic_value)

    def random_initialization(self, low=0.0, high=1.0):
        self.w = self.rng.uniform(low, high, size=self.w.shape)

    def zero_initialization(self):
        self.w.fill(0.0)

    def weights_initialization(self):
        if self.config.weight_init_choice == "optimistic":
            self.optimistic_initialization(self.config.optimistic_value)
        elif self.config.weight_init_choice == "random":
            self.random_initialization()
        elif self.config.weight_init_choice == "zero":
            self.zero_initialization()
        else:
            raise ValueError(
                f"Unknown weight initialization choice: {self.config.weight_init_choice}"
            )

    def q_values(self, state):
        """Return Q(s,a) vector in the same order as env.get_action_space()."""
        phi = self.feature_fn(state)
        return self.w @ phi  # shape: (num_actions,)

    def q_value(self, state, action):
        """Return scalar Q(s,action). Accepts action (not index)."""
        idx = self.action_to_index[action]
        phi = self.feature_fn(state)
        return float(self.w[idx] @ phi)

    # Epsilon-greedy that works for arbitrary action types
    def epsilon_greedy_action(self, state, epsilon):
        actions = self.actions
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)

        # exploitation using linear app
        q = self.q_values(state)  # vector of Qs aligned with `actions`
        max_q = np.max(q)
        best_idxs = np.flatnonzero(q == max_q)
        chosen_idx = int(self.rng.choice(best_idxs))
        return actions[chosen_idx]

    def run_episode(self, alpha: float, n_steps: int = 3):
        # For MountainCar: env.reset() returns the initial raw state
        state = self.env.reset()
        action = self.epsilon_greedy_action(state, self.config.epsilon)

        states = [state]
        actions = [action]
        rewards = (
            []
        )  # reward[t] = reward observed after taking action at time t (i.e., r_t)

        time = 0
        T = 1e9  # effectively infinity
        total_reward = 0.0

        while True:
            if time < T:
                next_state, reward, done, _ = self.env.step(state, action)
                rewards.append(reward)
                total_reward += reward
                self.num_steps += 1

                if done:
                    T = time + 1
                    states.append(next_state)
                    actions.append(None)  # placeholder for terminal
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
                # compute n-step return G for time tau
                G = 0.0
                # rewards indices: rewards[ tau ... min(tau+n_steps-1, T-1) ]
                upper_bound = min(
                    tau + n_steps, int(T) if T is not None else float("inf")
                )
                for i in range(tau, upper_bound):
                    G += (self.gamma ** (i - tau)) * rewards[i]

                # bootstrap if needed
                if tau + n_steps < T:
                    s_boot = states[tau + n_steps]
                    a_boot = actions[tau + n_steps]
                    # a_boot is not None because tau + n_steps < T
                    G += (self.gamma**n_steps) * self.q_value(s_boot, a_boot)

                # update weights for (s_tau, a_tau)
                if tau < T:
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    if a_tau is not None:
                        phi_tau = self.feature_fn(s_tau)
                        idx = self.action_to_index[a_tau]
                        q_hat = float(self.w[idx] @ phi_tau)
                        td_error = G - q_hat
                        # semi-gradient step
                        self.w[idx] += alpha * td_error * phi_tau

            if tau == T - 1:
                break

            time += 1

        # bookkeeping after episode
        self.num_episodes += 1
        self.pointers.append(self.num_steps)
        self.episode_rewards.append(total_reward)

        return total_reward

    def run(self, num_episodes: int = 10000, alpha: float = 0.1, n_steps: int = 3):
        self.reset()
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
        reward_metadata = []

        for _ in tqdm(
            range(num_times), desc=f"{num_times} Independent Runs", leave=False
        ):
            self.run(num_episodes, alpha, n_steps=n_steps)
            pointer_metadata.append(self.pointers.copy())
            reward_metadata.append(self.episode_rewards.copy())

        return (
            np.array(pointer_metadata, dtype=object),
            np.array(reward_metadata, dtype=object),
        )

    # convenience: get current greedy action index (not used for training)
    def greedy_action(self, state):
        q = self.q_values(state)
        best_idxs = np.flatnonzero(q == np.max(q))
        return self.actions[int(self.rng.choice(best_idxs))]


class MountainCarSARSAHelpers:

    def __init__(self):
        pass

    def _get_scaled_state(self, state, limits, num_tilings, num_tiles):
        range_ = limits[1] - limits[0]
        scaled_state = (state - limits[0]) / range_
        return scaled_state * (num_tiles - 1) * num_tilings

    def get_active_tiles(self, state, limits, num_tilings, num_tiles):
        scaled_state = self._get_scaled_state(state, limits, num_tilings, num_tiles)
        base_tile_index = np.floor(scaled_state).astype(int)
        active_tile_indices = []
        tiles_per_tiling = num_tiles ** len(state)
        for t in range(num_tilings):
            tiled_coords = base_tile_index - t
            current_index = 0
            for i, coord in enumerate(tiled_coords):
                stride = num_tiles**i
                current_index += (coord % (num_tiles - 1)) * stride
            global_index = t * tiles_per_tiling + current_index
            active_tile_indices.append(global_index)

        return np.array(active_tile_indices, dtype=int)

    def make_tile_coding_feature_fn(self, env, num_tilings=8, num_tiles=8):
        obs_space = env.env.observation_space  # Assumes gym/MDPBase structure
        limits = np.asarray([obs_space.low, obs_space.high], dtype=np.float64).T
        num_dimensions = len(env.reset())
        num_features = (num_tiles**num_dimensions) * num_tilings

        def feature_fn(state):
            """Returns the sparse feature vector for a given state."""
            active_indices = self.get_active_tiles(
                state, limits, num_tilings, num_tiles
            )
            phi = np.zeros(num_features, dtype=np.float64)
            phi[active_indices] = 1.0
            return phi

        return feature_fn, num_features
