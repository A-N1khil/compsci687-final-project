import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from dataclasses import dataclass
from mdp.gridworld.gridworld import GridWorldMDP


# defining config dataclass for hyperparameters

STEP_OUTPUT_LEN_WITH_INFO = 4

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
    


# defining small helper to convert state to numpy array
def to_numpy_state(state):
    """
    Converting arbitrary state to flat float32 numpy array.
    """
    if isinstance(state, np.ndarray):
        return state.astype(np.float32).reshape(-1)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32).reshape(-1)
    # fallback for scalar-like inputs
    return np.array([state], dtype=np.float32)


class DiscreteActorCriticNet(nn.Module):

    # building shared actor-critic network
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()

        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_size),nn.ReLU(),nn.Linear(hidden_size, hidden_size),nn.ReLU(),)

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    # running forward pass for a batch of states
    def forward_pass(self, state_tensor: torch.Tensor):
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)  # (D,) -> (1, D)

        features = self.shared(state_tensor)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


class DiscreteActorCriticAgent:

    # initialising actor-critic agent for discrete action spaces
    def __init__(self, state_dim, action_dim, config: ACConfig, device=None):
        self.cfg = config
        self.gamma = config.gamma
        self.entropy_coef = config.entropy_coef

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DiscreteActorCriticNet(state_dim=state_dim,action_dim=action_dim,hidden_size=config.hidden_size,).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)

    # evaluating state and returning policy logits and value
    def evaluate_state(self, state_np: np.ndarray):
        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        logits, value = self.net.forward_pass(state_tensor)
        return logits, value

    # sampling action from categorical policy
    def sample_action(self, logits: torch.Tensor):
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    # selecting action for a given state
    def select_action(self, state_np: np.ndarray):
        logits, value = self.evaluate_state(state_np)
        action_tensor, log_prob, entropy = self.sample_action(logits)

        # removing batch dimension
        action_index = int(action_tensor.item())
        value = value.squeeze(0)

        return action_index, log_prob, value, entropy

    # computing discounted returns with optional bootstrapping
    def compute_returns(self, rewards, last_value, done):
        returns = []
        G = 0.0 if done else last_value

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns

    # computing total loss for actor, critic, and entropy regularisation
    def compute_loss(self, log_probs, values, returns, entropies):
        values_tensor = torch.stack(values).squeeze(1)

        advantages = returns - values_tensor
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(torch.stack(log_probs) * adv_norm.detach()).mean()
        critic_loss = 0.5 * (advantages.pow(2)).mean()
        entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()

        total_loss = actor_loss + critic_loss - entropy_bonus
        return total_loss

    # running one episode on the given environment
    def run_episode(self, env):
        state = env.reset()
        state_np = to_numpy_state(state)

        log_probs = []
        values = []
        rewards = []
        entropies = []

        done = False

        for _ in range(self.cfg.max_steps_per_episode):
            action_idx, log_prob, value, entropy = self.select_action(state_np)

            step_out = env.step(action_idx)
            if len(step_out) == STEP_OUTPUT_LEN_WITH_INFO:
                next_state, reward, done, _ = step_out
            else:
                next_state, reward, done = step_out

            next_state_np = to_numpy_state(next_state)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            state_np = next_state_np

            if done:
                break

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np).float().to(self.device)
            _, last_value_tensor = self.net.forward_pass(state_tensor)
            last_value = float(last_value_tensor.squeeze(0).item())

        return log_probs, values, rewards, entropies, last_value, done

    # running full training loop across episodes
    def train(self, env, log_interval=10, debug_interval=50):
        episode_rewards = []
        best_reward = -1e9

        for episode in range(self.cfg.num_episodes):
            (log_probs,values,rewards,entropies,last_value,done,) = self.run_episode(env)

            returns = self.compute_returns(rewards, last_value, done)

            # applying very mild entropy decay over time
            self.entropy_coef = max(1e-6, self.entropy_coef * 0.995)

            loss = self.compute_loss(log_probs, values, returns, entropies)
            episode_return = float(sum(rewards))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            episode_rewards.append(episode_return)
            best_reward = max(best_reward, episode_return)

            if (episode + 1) % log_interval == 0:
                print(
                    f"Episode {episode+1}/{self.cfg.num_episodes}, "
                    f"Return: {episode_return:.2f}, "
                    f"Best so far: {best_reward:.2f}"
                )

            if episode % debug_interval == 0:
                with torch.no_grad():
                    eval_state = env.reset()
                    eval_state_np = to_numpy_state(eval_state)
                    eval_tensor = (
                        torch.from_numpy(eval_state_np)
                        .float()
                        .to(self.device)
                    )
                    logits, value_eval = self.net.forward_pass(eval_tensor)
                    dist = Categorical(logits=logits)
                    entropy_eval = dist.entropy().mean()

                print(f"\nEpisode {episode}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Value Estimate: {value_eval.item():.4f}")
                print(f"Policy logits: {logits.cpu().numpy()}")
                print(f"Entropy Mean: {entropy_eval.item():.4f}")
                print(f"Episode Return: {episode_return:.2f}\n")

        return episode_rewards, best_reward


# defining helper to infer action dimension from environment
def infer_action_dim(env):

    if hasattr(env, "action_space"):
        space = env.action_space
        if hasattr(space, "n"):
            return space.n
        try:
            return len(space)
        except TypeError:
            raise ValueError("Cannot infer action_dim: env.action_space has no .n and is not len()-able.")
    else:
        raise ValueError("Environment must expose action_space with either .n or be a list-like of actions.")


# running training for a given environment and config
def train_actor_critic(env, config: ACConfig, log_interval=10, debug_interval=50):

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # probing state dimension from a reset
    init_state = env.reset()
    state_dim = to_numpy_state(init_state).shape[0]

    # probing number of actions
    action_dim = infer_action_dim(env)

    agent = DiscreteActorCriticAgent(state_dim=state_dim,action_dim=action_dim,config=config,)

    episode_rewards, best_reward = agent.train(env,log_interval=log_interval,debug_interval=debug_interval,)

    return {
        "config": config,
        "episode_rewards": episode_rewards,
        "best_reward": best_reward,
    }


# GridWorld Wrapper for Discrete Actor-Critic

class GridWorldEnvWrapper:
    """
    Wrapping GridWorldMDP to expose a Gym-like API:
    - reset() -> state_vec
    - step(action_index) -> next_state_vec, reward, done, {}
    - action_space: int list [0..num_actions-1]
    """

    def __init__(self, max_steps: int = 60):
        # storing original MDP
        self.mdp = GridWorldMDP()
        self.max_steps = max_steps

        # caching sizes
        self.rows = self.mdp.rows
        self.cols = self.mdp.cols

        # mapping actions: index → name
        # ["up","down","left","right"] → [0,1,2,3]
        self.action_list = self.mdp.get_action_space()
        self.action_space = list(range(len(self.action_list)))

        # runtime variables
        self.current_state = None
        self.steps = 0

    # Reset and convert state to one-hot vector for network input
    def reset(self):
        self.current_state = self.mdp.reset()
        self.steps = 0
        return self.state_to_vector(self.current_state)

    # converting (row,col) → one-hot vector of length rows*cols
    def state_to_vector(self, state):
        r, c = state
        vec = np.zeros(self.rows * self.cols, dtype=np.float32)
        idx = r * self.cols + c
        vec[idx] = 1.0
        return vec

    # Step using discrete action index
    def step(self, action_idx):
        action_name = self.action_list[action_idx]
        next_state, reward, done = self.mdp.step(self.current_state, action_name)

        self.current_state = next_state
        self.steps += 1

        # enforce episode cutoff
        if self.steps >= self.max_steps:
            done = True

        return self.state_to_vector(next_state), reward, done, {}

