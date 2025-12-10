import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from dataclasses import dataclass
from mdp.gridworld.gridworld import GridWorldMDP
from mdp.cartpole.cartpole import CartPoleMDP
from torch.optim.lr_scheduler import CosineAnnealingLR  

STEP_OUTPUT_LEN_WITH_INFO = 4

# storing training configuration
@dataclass
class ACConfig:
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    lr: float = 1e-3
    entropy_coef: float = 0.01
    hidden_size: int = 128
    seed: int = 42
    label: str = "default"

# converting state to numpy vector
def to_numpy_state(state):
    if isinstance(state, np.ndarray):
        return state.astype(np.float32).reshape(-1)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32).reshape(-1)
    return np.array([state], dtype=np.float32)


class DiscreteActorCriticNet(nn.Module):

    # creating shared layers for policy and value networks

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),  # Remove LayerNorm
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),  # Remove LayerNorm
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        # Orthogonal init for better stability
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    # generating logits and value for given state batch
    def forward_pass(self, state_tensor):
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        features = self.shared(state_tensor)
        return self.policy_head(features), self.value_head(features)


class DiscreteActorCriticAgent:

    # initialising network and optimizer
    def __init__(self, state_dim, action_dim, config: ACConfig, device=None):
        self.cfg = config
        self.gamma = config.gamma
        self.entropy_coef = config.entropy_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DiscreteActorCriticNet(state_dim, action_dim, config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)

        # NEW: cosine LR scheduler over episodes
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.num_episodes
        )

    # evaluating a single state through network
    def evaluate_state(self, state_np):
        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        return self.net.forward_pass(state_tensor)

    # sampling an action and log probability
    def sample_action(self, logits):
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    # selecting action from current policy
    def select_action(self, state_np):
        logits, value = self.evaluate_state(state_np)
        action_tensor, log_prob, entropy = self.sample_action(logits)
        return int(action_tensor.item()), log_prob, value.squeeze(0), entropy

    # computing discounted returns
    def compute_returns(self, rewards, last_value, done):
        G = 0.0 if done else last_value
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)


    def compute_loss(self, log_probs, values, returns, entropies):
        values_tensor = torch.stack(values).squeeze(1)
        advantages = returns - values_tensor

        # Normalised advantages WITH clipping
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -5.0, 5.0)  # NEW: Clip extreme values

        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()

        # NEW: Adaptive critic weight
        critic_weight = 0.5 if len(log_probs) < 100 else 0.25
        critic_loss = critic_weight * advantages.pow(2).mean()

        entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()
        return actor_loss + critic_loss - entropy_bonus

    # running one full episode
    def run_episode(self, env):
        state_np = to_numpy_state(env.reset())
        log_probs, values, rewards, entropies = [], [], [], []
        done = False

        for _ in range(self.cfg.max_steps_per_episode):
            action_idx, log_prob, value, entropy = self.select_action(state_np)
            # CORRECTED: directly unpack from env.step()
            next_state, reward, done, _ = env.step(action_idx)
            state_np = to_numpy_state(next_state)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            if done:
                break

        with torch.no_grad():
            _, last_value_tensor = self.net.forward_pass(
                torch.from_numpy(state_np).float().to(self.device)
            )
        last_value = float(last_value_tensor.squeeze(0).item())
        return log_probs, values, rewards, entropies, last_value, done

    # training agent through episodes
    def train(self, env, log_interval=10, debug_interval=50):
        episode_rewards, episode_losses = [], []
        episode_values, episode_entropies = [], []
        best_reward = -1e9
        
        warmup_episodes = 100
        
        return_scale = 0.01

        for episode in range(self.cfg.num_episodes):
            log_probs, values, rewards, entropies, last_value, done = self.run_episode(env)


            
            # NEW: Scale down rewards (CartPole returns are large: 1 per step)
            scaled_rewards = [r * return_scale for r in rewards]

            # CHANGED: Use scaled rewards for returns calculation
            returns = self.compute_returns(scaled_rewards, last_value * return_scale, done)

            # CHANGED: Slower entropy annealing
            self.entropy_coef = max(1e-4, self.entropy_coef * 0.9995)  # 0.999 → 0.9995

            loss = self.compute_loss(log_probs, values, returns, entropies)

            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # Changed from 1.0
            
            self.optimizer.step()
            
            if episode >= warmup_episodes:
                self.scheduler.step()  
            
            if episode < warmup_episodes:
                warmup_factor = (episode + 1) / warmup_episodes
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg.lr * warmup_factor

            episode_return = float(sum(rewards))  # Report original unscaled return
            episode_rewards.append(episode_return)
            episode_losses.append(loss.item())
            best_reward = max(best_reward, episode_return)

            if episode % debug_interval == 0:
                with torch.no_grad():
                    eval_state = env.reset()
                    eval_logits, eval_value = self.net.forward_pass(
                        torch.from_numpy(to_numpy_state(eval_state)).float().to(self.device)
                    )
                    dist = Categorical(logits=eval_logits)
                episode_values.append(eval_value.item())
                episode_entropies.append(dist.entropy().mean().item())

            if (episode + 1) % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Episode {episode+1}/{self.cfg.num_episodes}, "
                    f"Return: {episode_return:.2f}, Best: {best_reward:.2f}, "
                    f"LR: {current_lr:.6f}, Entropy: {self.entropy_coef:.4f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "episode_values": episode_values,
            "episode_entropies": episode_entropies,
            "best_reward": best_reward,
            "config": self.cfg,
        }


# inferring action dimension from environment
def infer_action_dim(env):
    space = env.action_space
    if hasattr(space, "n"):
        return space.n
    return len(space)

# training agent with given configuration
def train_actor_critic(env, config: ACConfig, log_interval=10, debug_interval=50):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    state_dim = to_numpy_state(env.reset()).shape[0]
    action_dim = infer_action_dim(env)
    agent = DiscreteActorCriticAgent(state_dim, action_dim, config)
    result = agent.train(env, log_interval=log_interval, debug_interval=debug_interval)
    result["config"] = config
    return result


class GridWorldEnvWrapper:

    # initialising gridworld wrapper with max steps
    def __init__(self, max_steps: int = 60):
        self.mdp = GridWorldMDP()
        self.max_steps = max_steps
        self.rows, self.cols = self.mdp.rows, self.mdp.cols
        self.action_list = self.mdp.get_action_space()
        self.action_space = list(range(len(self.action_list)))
        self.current_state = None
        self.steps = 0

    # resetting environment to start state
    def reset(self):
        self.current_state = self.mdp.reset()
        self.steps = 0
        return self.state_to_vector(self.current_state)

    # converting state into one-hot encoding
    def state_to_vector(self, state):
        r, c = state
        vec = np.zeros(self.rows * self.cols, dtype=np.float32)
        vec[r * self.cols + c] = 1.0
        return vec

    # stepping environment using action index
    def step(self, action_idx):
        next_state, reward, done = self.mdp.step(
            self.current_state,
            self.action_list[action_idx]
        )
        self.current_state = next_state
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        return self.state_to_vector(next_state), reward, done, {}
    

# CartPole Wrapper for Discrete Actor-Critic

class CartPoleEnvWrapper:
    def __init__(self, max_steps: int = 500, seed: int = 42):
        from mdp.cartpole.cartpole import CartPoleMDP
        self.mdp = CartPoleMDP(seed=seed)
        self.max_steps = max_steps
        self.steps = 0
        self.current_state = None

        # discrete action list
        self.action_list = self.mdp.get_action_space()
        self.action_space = list(range(len(self.action_list)))

    def reset(self):
        state = self.mdp.reset()
        self.current_state = state
        self.steps = 0
        return state.astype(np.float32)

    def step(self, action_idx):
        # stepping CartPole MDP with current state and action
        next_state, reward, done = self.mdp.step(self.current_state, action_idx)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        # updating current state for next step
        self.current_state = next_state

        return next_state.astype(np.float32), float(reward), done, {}