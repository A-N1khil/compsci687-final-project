# COMPSCI 687: Reinforcement Learning - Final Project

## Overview
This repository contains the code and documentation for the final project of COMPSCI 687: Reinforcement Learning.

## MDPs
- [gridworld.py](mdp/gridworld) — Environment definition for the Gridworld MDP (state transitions, rewards, and utilities) used by multiple algorithms and notebooks.
- [cats_vs_monsters.py](mdp/cats_v_monsters) — Environment definition for the Cats vs Monsters MDP
- [mountain_car.py](mdp/mountaincar) — Environment definition for the Mountain Car MDP
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