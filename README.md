# COMPSCI 687: Reinforcement Learning - Final Project

## Overview

This repository contains the code and documentation for the final project of COMPSCI 687: Reinforcement Learning.

## MDPs

- [gridworld](mdp/gridworld) — Environment definition for the Gridworld MDP (state transitions, rewards, and utilities) used by multiple algorithms and notebooks.
- [cats_vs_monsters](mdp/cats_v_monsters) — Environment definition for the Cats vs Monsters MDP
- [mountain_car](mdp/mountaincar) — Environment definition for the Mountain Car MDP
- [cartpole.py](mdp/cartpole) — Environment definition for the CartPole MDP

## Algorithms

- [reinforce_baseline](algorithms/reinforce_baseline) — PyTorch REINFORCE with baseline implementation (agent, policy network, baseline estimator, and trajectory utilities) used for episodic policy-gradient experiments.
- [semi_gradient_nstep_sarsa](algorithms/sgns_sarsa) - Semi-Gradient $n$-step Sarsa implementation for function approximation
- [actor_critic](algorithms/actor_critic) — Actor-Critic implementation
- [value_iteration](algorithms/value_iteration) — Value Iteration implementation for tabular environments

## Running the code

### Setup

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Checking the outputs in Notebooks
You can run the following Jupyter notebooks to see the results of various algorithms on different environments:
- [cats_vs_monsters_ac.ipynb](cats_vs_monsters_ac.ipynb) — Actor-Critic experiments on the Cats vs Monsters environment
- [reinforce_baseline.ipynb](reinforce_baseline.ipynb) — Runs REINFORCE-with-baseline algorithm across supported environments
- [value_iteration_policies.ipynb](value_iteration_policies.ipynb) — Solves the tabular environments via value iteration and exports baseline/optimal policies for comparison
- [sgns_sarsa_cvm.ipynb](sgnstep_sarsa_cvm.ipynb) — Semi-Gradient n-step Sarsa experiments on Cats vs Monsters environment
- [sgns_sarsa_gridworld.ipynb](sgnstep_sarsa_baseline.ipynb) — Semi-Gradient n-step Sarsa experiments on Gridworld environment
- [sgns_sarsa_mc.ipynb](sgnstep_sarsa_mc.ipynb) — Semi-Gradient n-step Sarsa experiments on Mountain Car environment
- [sgns_sarsa_cp.ipynb](sgnstep_sarsa_cp.ipynb) — Semi-Gradient n-step Sarsa experiments on CartPole environment

### Python scripts

#### Reinforce with Baseline

```bash
python reinforce_baseline.py
```

#### Semi-Gradient n-step Sarsa

```bash
# Cats vs Monsters
python sgns_sarsa_cvm.py
# Gridworld
python sgns_sarsa_gridworld.py
# Mountain Car
python sgns_sarsa_mc.py
# CartPole
python sgns_sarsa_cp.py
```
