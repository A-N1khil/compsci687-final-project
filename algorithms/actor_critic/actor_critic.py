import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

# MountainCar state normalization ranges
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07

def normalize_state(state):
    state = state.copy()
    state[0] = (state[0] - POS_MIN) / (POS_MAX - POS_MIN)
    state[1] = (state[1] - VEL_MIN) / (VEL_MAX - VEL_MIN)
    return state

def to_numpy(state):
    # Convert state to a numpy array for consistent handling
    # Needed for gridworld and CatvsMonsters as they use tuples for states

    if isinstance(state, np.ndarray):
        return state.astype(np.float32)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32)
    
    #fallback incase of unexpected inputs
    return np.array([state], dtype=np.float32)

class ActorCriticNN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):

        super().__init__()

        # Shared feature extractor
        # Edit 1:
        # Originally used separate Actor and Critic networks, which increased parameters and slowed training.
        # Switched to a shared feature encoder so both heads learn from the same state representation.
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),)

        # Actor: produces logits for each action
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic: predicts scalar V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward_pass(self, state):
        #Converts state (numpy or torch) into logits + value.
        #If input is 1D → add batch dimension.

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (state_dim,) → (1,state_dim)

        features = self.shared(state)
        logits = self.actor(features)      # for action sampling
        value = self.critic(features)      # state-value estimate
            
        return logits, value

class ActorCriticAgent:

    def __init__(self, mdp, state_dim, action_dim, gamma=0.99, lr=1e-3, entropy_coef=1e-3, device=None):
        # saving env reference
        self.mdp = mdp
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # choose device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # create the neural network brain
        # self.net = ActorCriticNN(state_dim, action_dim).to(self.device)
        self.net = ActorCriticNN(state_dim, action_dim, hidden_dim=128).to(self.device)


        # Adam optimizer for all parameters
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)


    def evaluate_state(self, state_np):
        # Convert numpy to tensor and forward pass
        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        logits, value = self.net.forward_pass(state_tensor)
        return logits, value


    def sample_action(self, logits):
        # Create distribution over actions and sample from it
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy


    def select_action(self, state_np):
        logits, value = self.evaluate_state(state_np)
        action, log_prob, entropy = self.sample_action(logits)
        value = value.squeeze(0)
        return action, log_prob, value, entropy

    def compute_returns(self, rewards):
        # Compute discounted returns
        returns = []
        G = 0  # running sum
        for r in reversed(rewards): # iterate backwards to account for future rewards first
            G = r + self.gamma * G
            returns.insert(0, G)  # prepend so order matches steps
        return torch.tensor(returns, dtype=torch.float32).to(self.device)


    def run_episode(self, max_steps=200):

        state = normalize_state(to_numpy(self.mdp.reset())) 

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for _ in range(max_steps):
            # select action from current policy
            action, log_prob, value, entropy = self.select_action(state)

            # interact with environment
            next_state, reward, done, _ = self.mdp.step(action)
            shaped_reward = reward + 0.1 * (next_state[0] - state[0])

            # store info for update
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(shaped_reward)
            entropies.append(entropy)

            state = normalize_state(to_numpy(next_state))

            if done:
                break

        return log_probs, values, rewards, entropies
    
    
    # def compute_loss(self, log_probs, values, rewards, entropies):
        
    #     returns = self.compute_returns(rewards)

    #     values = torch.stack(values).squeeze(1)
    #     advantages = returns - values

    #     actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    #     critic_loss = advantages.pow(2).mean()
    #     entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()
    #     total_loss = actor_loss + critic_loss - entropy_bonus
    #     episode_reward = sum(rewards)  # performance metric


    #     return total_loss, episode_reward

    def compute_loss(self, log_probs, values, rewards, entropies):
        returns = self.compute_returns(rewards)
        values = torch.stack(values).squeeze(1)

        advantages = returns - values
        advantages = advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 🚀 normalize

        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_bonus = self.entropy_coef * torch.stack(entropies).mean()

        return actor_loss + critic_loss - entropy_bonus

    

    def train(self, num_episodes=500, max_steps=200, log_interval=10):
        all_rewards = []

        for episode in range(num_episodes):
            log_probs, values, rewards, entropies = self.run_episode(max_steps)
            
            # compute update loss and episode return
            loss = self.compute_loss(log_probs, values, rewards, entropies)
            episode_reward = sum(rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_rewards.append(episode_reward)

            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")

        return all_rewards


  

