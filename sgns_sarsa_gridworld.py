#!/usr/bin/env python
# coding: utf-8

# # Semi Gradient n-step Sarsa

# In[1]:


from algorithms.sgns_sarsa.n_step_sarsa import SGNStepSARSA, Config
from mdp.base_config import BaseConfig
import numpy as np
import matplotlib.pyplot as plt
from mdp.gridworld.gridworld import GridWorldMDP
from algorithms.value_iteration.value_iteration import ValueIteration
from mdp.mdp_utils import display_value_function, display_policy, max_norm_error
from itertools import product
from tqdm.notebook import tqdm

base_config = BaseConfig(42)


# ## Gridworld

# ### Find the optimal value function and policy using Value Iteration

# In[2]:


optimal_mdp = GridWorldMDP()
optimal_vi = ValueIteration(mdp=optimal_mdp)
optimal_rh = optimal_vi.run_value_iteration(gamma=0.9)
optimal_vf, optimal_policy = optimal_rh[len(optimal_rh)]
print("Optimal Value Function:")
display_value_function(optimal_mdp, optimal_vf)
print("Optimal Policy:")
display_policy(optimal_mdp, optimal_policy)


# In[3]:


gw = GridWorldMDP()
config = Config(q_init_choice="equal", optimistic_value=10, epsilon=0.1)
sarsa = SGNStepSARSA(gw, optimal_vf=optimal_vf, gamma=0.9)


# In[4]:


reward = sarsa.run_episode(alpha=0.01, n_steps=2)
print(f"Total reward in one episode: {reward}")


# In[5]:


points_metadata, mse_metadata, _ = sarsa.run_n_times(
    num_times=20, num_episodes=3000, alpha=0.01, n_steps=3
)


# In[6]:


print(points_metadata.shape)
mean_steps = np.mean(points_metadata, axis=0)
print(mean_steps.shape)


# In[7]:


episodes = np.arange(1, mean_steps.shape[0] + 1)
plt.figure(figsize=(10, 6))
for run in [0, 4, 9, 14, 19]:
    plt.plot(points_metadata[run], episodes, label=f"Episode {run+1}")
plt.xlabel("Total Actions Taken")
plt.ylabel("Episodes Completed")
plt.title("Learning Curves for 20 runs")
plt.grid(True)
plt.legend()
plt.show()


# In[8]:


plt.figure(figsize=(10, 6))
episodes = np.arange(1, mean_steps.shape[0] + 1)
plt.plot(mean_steps, episodes)
plt.xlabel("Total Actions Taken")
plt.ylabel("Episodes Completed")
plt.title("Average Learning Curve (20 runs)")
plt.grid(True)
plt.show()


# In[9]:


episode_lengths = np.diff(mean_steps)
plt.plot(episode_lengths)
plt.show()


# In[10]:


print(mse_metadata.shape)
mean_mse = np.mean(mse_metadata, axis=0)
print(mean_mse.shape)


# In[11]:


plt.figure(figsize=(10, 6))
episodes = np.arange(1, mse_metadata.shape[1] + 1)
plt.plot(episodes, mean_mse)
plt.axhline(y=0, color="g", label="0 MSE")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.title("Average MSE Learning Curve (20 runs)")
plt.grid(True)
plt.show()


# In[12]:


alpha_values = [0.001, 0.003, 0.007, 0.01]
n_step_values = [1, 3, 5]
gw = GridWorldMDP()
config = Config(q_init_choice="equal", epsilon=0.1)
results = {}
hp_combos = list(product(alpha_values, n_step_values))
for alpha, n_steps in tqdm(hp_combos, desc="Running hyperparameter combinations"):
    print(f"Running for alpha={alpha}, n_steps={n_steps}")
    sarsa = SGNStepSARSA(gw, optimal_vf=optimal_vf, gamma=0.9)
    points_metadata, mse_metadata, _ = sarsa.run_n_times(
        num_times=10, num_episodes=3000, alpha=alpha, n_steps=n_steps
    )
    mean_steps = np.mean(points_metadata, axis=0)
    mean_mse = np.mean(mse_metadata, axis=0)
    results[(alpha, n_steps)] = {"mean_steps": mean_steps, "mean_mse": mean_mse}


# In[13]:


# Print MSE Stats - mean, median and std for each alpha and n_steps as tabulate
from tabulate import tabulate

mse_stats = []
for (alpha, n_steps), data in results.items():
    mean_mse = np.mean(data["mean_mse"])
    median_mse = np.median(data["mean_mse"])
    std_mse = np.std(data["mean_mse"])
    max_mse = np.max(data["mean_mse"])
    min_mse = np.min(data["mean_mse"])
    mse_stats.append([alpha, n_steps, mean_mse, median_mse, std_mse, max_mse, min_mse])
print(
    tabulate(
        mse_stats,
        headers=[
            "Alpha",
            "N Steps",
            "Mean MSE",
            "Median MSE",
            "Std MSE",
            "Max MSE",
            "Min MSE",
        ],
        tablefmt="grid",
    )
)


# In[14]:


# Plot MSE for each hyperparameter combination
plt.figure(figsize=(14, 8))
for (alpha, n_steps), data in results.items():
    episodes = np.arange(1, data["mean_mse"].shape[0] + 1)
    plt.plot(episodes, data["mean_mse"], label=f"alpha={alpha}, n_steps={n_steps}")
plt.axhline(y=0, color="g", label="0 MSE")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.show()


# In[15]:


# Considering only alpha=0.01, 0.05 and n_steps=1, 3, 5
alpha_vals_fine = [0.001, 0.003]
n_steps_vals_fine = [1, 3, 5]
# Plotting MSE for fine-tuned hyperparameters
plt.figure(figsize=(10, 6))
for (alpha, n_steps), data in results.items():
    if alpha in alpha_vals_fine and n_steps in n_steps_vals_fine:
        episodes = np.arange(1, data["mean_mse"].shape[0] + 1)
        plt.plot(episodes, data["mean_mse"], label=f"alpha={alpha}, n_steps={n_steps}")
plt.axhline(y=0, color="g", label="0 MSE")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.title("MSE for Fine-tuned Hyperparameters")
plt.grid(True)
plt.legend()
plt.show()


# ### Plotting Mean Steps

# In[16]:


alpha_vals_fine = [0.001, 0.003]
n_steps_vals_fine = [1, 3, 5]
hp_combos_fine = list(product(alpha_vals_fine, n_steps_vals_fine))
# Plotting MSE for fine-tuned hyperparameters
for alpha, n_steps in hp_combos_fine:
    if (alpha, n_steps) in results:
        data = results[(alpha, n_steps)]
        episodes = np.arange(1, data["mean_steps"].shape[0] + 1)
        plt.plot(
            data["mean_steps"], episodes, label=f"alpha={alpha}, n_steps={n_steps}"
        )
plt.xlabel("Cumulative Steps")
plt.ylabel("Episodes")
plt.title("Mean Steps for Fine-tuned Hyperparameters")
plt.grid(True)
plt.legend()
plt.show()


# In[17]:


best_alpha, best_n_steps = min(
    results.keys(), key=lambda k: np.mean(results[k]["mean_mse"])
)
print(f"Best hyperparameters: alpha={best_alpha}, n_steps={best_n_steps}")


# In[18]:


# Testing epsilon values
epsilon_values = [0.1, 0.3, 0.7, 0.9, 1.0]
gw = GridWorldMDP()
results_eps = {}
for epsilon in tqdm(epsilon_values, desc="Running for different epsilon values"):
    print(f"Running for epsilon={epsilon}")
    config = Config(q_init_choice="zeroes", epsilon=epsilon)
    sarsa = SGNStepSARSA(gw, optimal_vf=optimal_vf, gamma=0.9, config=config)
    points_metadata, mse_metadata, _ = sarsa.run_n_times(
        num_times=10, num_episodes=3000, alpha=best_alpha, n_steps=best_n_steps
    )
    mean_steps = np.mean(points_metadata, axis=0)
    mean_mse = np.mean(mse_metadata, axis=0)
    results_eps[epsilon] = {"mean_steps": mean_steps, "mean_mse": mean_mse}


# In[19]:


# Plotting the results
plt.figure(figsize=(12, 6))
for epsilon, data in results_eps.items():
    episodes = np.arange(1, data["mean_mse"].shape[0] + 1)
    plt.plot(episodes, data["mean_mse"], label=f"epsilon={epsilon}")
plt.axhline(y=0, color="g", label="0 MSE")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.title("MSE Learning Curve for Different Epsilon Values")
plt.grid(True)
plt.legend()
plt.show()


# In[20]:


config = Config(q_init_choice="zeroes", epsilon=0.1)
sarsa = SGNStepSARSA(gw, optimal_vf=optimal_vf, gamma=0.9, config=config)
points_metadata, mse_metadata, vf_data = sarsa.run_n_times(
    num_times=20, num_episodes=3000, alpha=best_alpha, n_steps=best_n_steps
)
mean_steps = np.mean(points_metadata, axis=0)
mean_mse = np.mean(mse_metadata, axis=0)
print(f"Mean Steps: {mean_steps.shape}, Mean MSE: {mean_mse.shape}")
plt.figure(figsize=(10, 6))
episodes = np.arange(1, mean_steps.shape[0] + 1)
plt.plot(mean_steps, episodes)
plt.xlabel("Total Actions Taken")
plt.ylabel("Episodes Completed")
plt.title("Average Learning Curve (20 runs)")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
episodes = np.arange(1, mean_mse.shape[0] + 1)
plt.plot(episodes, mean_mse)
plt.axhline(y=0, color="g", label="0 MSE")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.title("Average MSE Learning Curve (20 runs)")
plt.grid(True)
plt.show()


# In[21]:


print(vf_data.shape)
average_vf = {}
for vf in vf_data:
    for state in vf:
        if state in average_vf:
            average_vf[state] += vf[state]
        else:
            average_vf[state] = vf[state]

for state in average_vf:
    average_vf[state] /= vf_data.shape[0]

display_value_function(optimal_mdp, average_vf)
max_error = max_norm_error(average_vf, optimal_vf)
print(f"Max norm error: {max_error}")


# In[22]:


episode_lengths = np.diff(mean_steps)
plt.plot(episode_lengths)
plt.xlabel("Episodes")
plt.ylabel("Episode Lengths")
plt.title("Episode Lengths over Episodes")
plt.show()


# In[23]:


policy = sarsa.get_greedy_policy()
display_policy(optimal_mdp, policy)


# In[26]:


rw_metadata = sarsa.run_n_times(
    num_times=20, num_episodes=5000, alpha=0.01, n_steps=3, get_reward_metadata=True
)


# In[27]:


print(rw_metadata.shape)

# rw_metadata shape should be (num_runs, num_episodes)
mean_rewards = np.mean(rw_metadata, axis=0)
print(mean_rewards.shape)

# smoothing window
window = 50
kernel = np.ones(window) / window
smoothed_mean_rewards = np.convolve(mean_rewards, kernel, mode="valid")

# Episode indices
episodes_raw = np.arange(1, len(mean_rewards) + 1)  # full length
episodes_smooth = np.arange(window, len(mean_rewards) + 1)  # aligned with 'valid'

plt.figure(figsize=(10, 6))

# Plot raw rewards
plt.plot(episodes_raw, mean_rewards, alpha=0.3, label="Mean Reward (raw)")

# Plot smoothed rewards
plt.plot(
    episodes_smooth,
    smoothed_mean_rewards,
    color="blue",
    label=f"Smoothed (window={window})",
)

plt.xlabel("Episodes")
plt.ylabel("Mean Discounted Reward")
plt.title("Mean Discounted Reward per Episode (20 runs)")
plt.grid(True)
plt.legend()
plt.show()
