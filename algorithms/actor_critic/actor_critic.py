import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch import optim

# defining MountainCar state normalization ranges
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07


def normalize_state(state):
    # normalizing position and velocity to [0,1] to help stable learning
    state = state.copy()
    state[0] = (state[0] - POS_MIN) / (POS_MAX - POS_MIN)
    state[1] = (state[1] - VEL_MIN) / (VEL_MAX - VEL_MIN)
    return state


def to_numpy(state):
    # converting state to numpy for consistent handling across MDPs
    if isinstance(state, np.ndarray):
        return state.astype(np.float32)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32)

    # handling any unexpected scalar-like inputs
    return np.array([state], dtype=np.float32)


class ActorCriticNN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # building shared feature extractor for both actor and critic
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: producing logits for each action
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic: predicting scalar V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward_pass(self, state):
        # converting incoming state (numpy or torch) to logits and value
        # adding batch dimension when state is 1D

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (state_dim,) → (1, state_dim)

        features = self.shared(state)
        logits = self.actor(features)   # for action sampling
        value = self.critic(features)   # state-value estimate

        return logits, value


class ActorCriticAgent:

    def __init__(
        self,
        mdp,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        entropy_coef=1e-3,
        use_reward_shaping=False,
        device=None,
    ):
        # saving MDP reference and hyperparameters
        self.mdp = mdp
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_reward_shaping = use_reward_shaping

        # selecting device (GPU if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # creating the shared actor-critic network
        # giving slightly larger hidden size by default
        self.net = ActorCriticNN(state_dim, action_dim, hidden_dim=128).to(self.device)

        # using Adam optimizer for all network parameters
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    # -------- policy / value utilities --------

    def evaluate_state(self, state_np):
        # converting numpy state to tensor and running forward pass
        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        logits, value = self.net.forward_pass(state_tensor)
        return logits, value

    def sample_action(self, logits):
        # creating categorical distribution over actions and sampling from it
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def select_action(self, state_np):
        # selecting action and returning everything needed for learning step
        logits, value = self.evaluate_state(state_np)
        action, log_prob, entropy = self.sample_action(logits)
        value = value.squeeze(0)  # removing batch dimension from V(s)
        return action, log_prob, value, entropy

    # -------- return / advantage utilities --------

    def compute_returns(self, rewards, last_value, done):
        # computing discounted returns with optional bootstrap
        returns = []
        G = 0.0 if done else last_value  # bootstrap if we hit time limit instead of true terminal

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def compute_loss(self, log_probs, values, returns, entropies):
        # stacking values and computing normalized advantages
        values = torch.stack(values).squeeze(1)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # actor: policy gradient with advantage
        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        # critic: regression to returns
        critic_loss = advantages.pow(2).mean()
        # exploration bonus through entropy
        entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()

        return actor_loss + critic_loss - entropy_bonus

    # -------- reward shaping just for MountainCar --------

    def shape_reward(self, state_np, next_state_np, done):
        """
        adding an extra signal that encourages climbing higher and moving faster.
        kept modest so raw env return (-200, -150, ...) still makes sense.
        """
        if not self.use_reward_shaping:
            return 0.0

        # assuming MountainCar-style state = [position, velocity]
        pos = float(next_state_np[0])
        vel = float(next_state_np[1])

        # height bonus: higher than left valley → small positive signal
        height_term = 0.1 * (pos - POS_MIN)  # in [0, ~0.18]

        # speed bonus: encourage swinging with larger |v|
        speed_term = 0.5 * abs(vel)          # in [0, ~0.035]

        bonus = height_term + speed_term

        # big extra bonus only when actually reaching the goal
        goal_pos = getattr(self.mdp, "goal_pos", 0.5)
        if done and pos >= goal_pos:
            bonus += 100.0

        return bonus

    # -------- episode rollouts --------

    def run_episode(self, max_steps=200):
        # resetting environment and preparing bookkeeping
        raw_state = self.mdp.reset()
        state = normalize_state(to_numpy(raw_state))

        log_probs, values, rewards_shaped, entropies = [], [], [], []

        episode_return_raw = 0.0  # sum of original env rewards (for logging)

        done = False
        for _ in range(max_steps):
            # selecting action from current policy
            action, log_prob, value, entropy = self.select_action(state)

            # stepping in environment
            next_state_raw, reward_raw, done, _ = self.mdp.step(action)
            episode_return_raw += reward_raw

            # computing shaped reward for learning
            shaped = reward_raw + self.shape_reward(state, next_state_raw, done)

            # storing experience
            log_probs.append(log_prob)
            values.append(value)
            rewards_shaped.append(shaped)
            entropies.append(entropy)

            # moving to next state
            state = normalize_state(to_numpy(next_state_raw))

            if done:
                break

        # estimating value of final state for bootstrapping
        with torch.no_grad():
            last_value = self.net.forward_pass(
                torch.from_numpy(state).float().to(self.device)
            )[1].item()

        return log_probs, values, rewards_shaped, entropies, last_value, done, episode_return_raw

    # -------- training loop --------

    def train(self, num_episodes=500, max_steps=200, log_interval=10):
        # running training loop across multiple episodes
        all_raw_returns = []

        for episode in range(num_episodes):
            (
                log_probs,
                values,
                rewards_shaped,
                entropies,
                last_value,
                done,
                raw_return,
            ) = self.run_episode(max_steps)

            # computing returns from shaped rewards
            returns = self.compute_returns(rewards_shaped, last_value, done)

            # computing final loss
            loss = self.compute_loss(log_probs, values, returns, entropies)

            # gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_raw_returns.append(raw_return)

            # progress logging using RAW env return (around -200 etc)
            if (episode + 1) % log_interval == 0:
                print(
                    f"Episode {episode+1}/{num_episodes}, "
                    f"Raw Reward: {raw_return:.2f}, "
                    f"Shaped Return: {sum(rewards_shaped):.2f}"
                )

            # occasional debug snapshot
            if episode % 20 == 0:
                with torch.no_grad():
                    test_state = normalize_state(to_numpy(self.mdp.reset()))
                    test_state_t = torch.from_numpy(test_state).float().unsqueeze(0).to(self.device)
                    test_logits, test_value = self.net.forward_pass(test_state_t)
                    test_probs = torch.softmax(test_logits, dim=1)

                print(f"\n[DEBUG] Episode {episode}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Value Estimate: {test_value.item():.4f}")
                print(f"Action Probabilities: {test_probs.cpu().numpy()}")
                print(f"Entropy Mean: {torch.stack(entropies).mean().item():.4f}")
                print(f"Raw Reward: {raw_return:.2f}")
                print(f"Shaped Return: {sum(rewards_shaped):.2f}\n")

        return all_raw_returns
