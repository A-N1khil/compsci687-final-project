import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# MountainCarContinuous state ranges
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07


def normalize_state(state):
    """
    Normalizing position and velocity approximately to [0, 1]
    to help stable learning on continuous Mountain Car.
    """
    state = np.array(state, dtype=np.float32).copy()
    state[0] = (state[0] - POS_MIN) / (POS_MAX - POS_MIN)
    state[1] = (state[1] - VEL_MIN) / (VEL_MAX - VEL_MIN)
    return state


def to_numpy(state):
    """
    Converting state to float32 numpy array for consistent handling across MDPs.
    """
    if isinstance(state, np.ndarray):
        return state.astype(np.float32)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32)
    # fallback for scalar-like inputs
    return np.array([state], dtype=np.float32)


class ActorCriticContinuousNN(nn.Module):
    """
    Shared actor-critic network for continuous action spaces.
    - shared trunk
    - actor head outputs mean (mu)
    - log_std is a learned parameter (state-independent)
    - critic head outputs V(s)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor: mean of Gaussian policy
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        # global log_std parameter (one per action dimension), start at std ≈ 1
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic: scalar state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward_pass(self, state):
        """
        Forward pass handling both numpy and tensor input, returning:
        - mu: mean of the policy
        - std: std dev of the policy
        - value: V(s)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (state_dim,) → (1, state_dim)

        features = self.shared(state)
        mu = self.mu_head(features)
        std = self.log_std.exp().expand_as(mu)
        value = self.value_head(features)

        return mu, std, value


class ContinuousActorCriticAgent:
    """
    Actor-Critic agent for continuous action spaces (e.g., MountainCarContinuous-v0).
    """

    def __init__(
        self,
        mdp,
        state_dim,
        action_dim,
        gamma=0.99,
        # lr=3e-4,
        lr = 1e-4,
        entropy_coef=1e-3,
        device=None,
    ):
        # saving MDP reference and hyperparameters
        self.mdp = mdp
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # selecting device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # creating actor-critic network
        self.net = ActorCriticContinuousNN(
            state_dim, action_dim, hidden_dim=128
        ).to(self.device)

        # Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # action bounds (continuous)
        low = mdp.get_action_space().low
        high = mdp.get_action_space().high
        self.action_low = float(low[0])
        self.action_high = float(high[0])

    def evaluate_state(self, state_np):
        """
        Running forward pass for a numpy state on the correct device.
        """
        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        mu, std, value = self.net.forward_pass(state_tensor)
        return mu, std, value

    def sample_action(self, mu, std):
        """
        Sampling a continuous action from Normal(mu, std) and computing:
        - clipped action (respecting env bounds)
        - log_prob
        - entropy
        """
        dist = Normal(mu, std)
        action = dist.sample()                      # no tanh, plain Gaussian
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # clipping to env bounds (keeps gradients approx even though
        # the true distribution is unbounded)
        action_clipped = torch.clamp(action, self.action_low, self.action_high)

        return action_clipped, log_prob, entropy

    def select_action(self, state_np):
        """
        Selecting action for a given state and returning everything needed
        for the learning step: action (numpy), log_prob, value, entropy.
        """
        mu, std, value = self.evaluate_state(state_np)
        action_tensor, log_prob, entropy = self.sample_action(mu, std)

        # remove batch dimension
        action_scalar = action_tensor.squeeze(0).detach().cpu().numpy()[0]
        value = value.squeeze(0)

        return action_scalar, log_prob, value, entropy

    def compute_returns(self, rewards, last_value, done):
        """
        Computing discounted returns G_t.
        If episode ended by time limit instead of reaching goal, we bootstrap
        from V(s_T) (last_value).
        """
        returns = []
        G = 0.0 if done else last_value

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def compute_loss(self, log_probs, values, returns, entropies):
        """
        Computing total loss = actor + critic - entropy_bonus.
        """
        values = torch.stack(values).squeeze(1)

        # advantage: detach for actor only
        advantages = returns - values
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(torch.stack(log_probs) * adv_norm.detach()).mean()
        critic_loss = 0.5 * (advantages.pow(2)).mean()
        entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()

        return actor_loss + critic_loss - entropy_bonus

    def run_episode(self, max_steps=1000):
        """
        Running one full episode and collecting trajectories for update.
        """
        state = normalize_state(to_numpy(self.mdp.reset()))

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for _ in range(max_steps):
            action, log_prob, value, entropy = self.select_action(state)
            next_state, reward, done, _ = self.mdp.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            state = normalize_state(to_numpy(next_state))

            if done:
                break

        # final value for bootstrapping when episode is cut by time limit
        with torch.no_grad():
            _, _, last_value_tensor = self.net.forward_pass(
                torch.from_numpy(state).float().to(self.device)
            )
        last_value = float(last_value_tensor.squeeze(0).item())

        return log_probs, values, rewards, entropies, last_value, done

    def train(self, num_episodes=1200, max_steps=1000, log_interval=10):
        """
        Main training loop across episodes.
        """
        all_raw_rewards = []

        for episode in range(num_episodes):
            (
                log_probs,
                values,
                rewards,
                entropies,
                last_value,
                done,
            ) = self.run_episode(max_steps)

            returns = self.compute_returns(rewards, last_value, done)

            # very mild entropy decay (optional, matches earlier runs)
            self.entropy_coef = max(1e-6, self.entropy_coef * 0.995)

            loss = self.compute_loss(log_probs, values, returns, entropies)
            episode_raw_reward = float(sum(rewards))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            all_raw_rewards.append(episode_raw_reward)

            if (episode + 1) % log_interval == 0:
                print(
                    f"Episode {episode+1}/{num_episodes}, "
                    f"Raw Reward: {episode_raw_reward:.2f}"
                )

            if episode % 20 == 0:
                with torch.no_grad():
                    test_state = normalize_state(self.mdp.reset())
                    test_state_t = (
                        torch.from_numpy(test_state)
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    mu, std, test_value = self.net.forward_pass(test_state_t)
                    test_dist = Normal(mu, std)
                    test_entropy = test_dist.entropy().sum(dim=-1).mean()

                print(f"\n[DEBUG] Episode {episode}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Value Estimate: {test_value.item():.4f}")
                print(f"Mu (mean action): {mu.cpu().numpy()}")
                print(f"Std (action): {std.cpu().numpy()}")
                print(f"Entropy Mean: {test_entropy.item():.4f}")
                print(f"Raw Reward: {episode_raw_reward:.2f}\n")

        return all_raw_rewards
