## Overview

- **`mdp/gridworld/`** — Environment definition for the Gridworld MDP (state transitions, rewards, and utilities) used by multiple algorithms and notebooks.
- **`algorithms/reinforce_baseline/`** — PyTorch REINFORCE with baseline implementation (agent, policy network, baseline estimator, and trajectory utilities) used for episodic policy-gradient experiments.
- **`cats_vs_monsters_ac.ipynb`** — Actor-Critic experiments on the Cats vs Monsters environment; runs training/evaluation loops and visualizes learning curves.
- **`reinforce_baseline.ipynb`** — Runs REINFORCE-with-baseline algorithm across supported environments, including training runs and metrics logging.
- **`value_iteration_policies.ipynb`** — Solves the tabular environments (Gridworld, Cats vs Monsters, MountainCar, etc.) via value iteration and exports baseline/optimal policies for comparison.
- **`reinforce_logs/`** - This folder has the episodic data of the environments

