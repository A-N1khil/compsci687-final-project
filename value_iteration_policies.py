#!/usr/bin/env python
# coding: utf-8

# # Value Iteration Policies
# 
# Run classic Value Iteration on the supported MDPs and serialize the resulting optimal policies under `policies/`.

# In[ ]:


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from algorithms.value_iteration.value_iteration import ValueIteration
from mdp.gridworld.gridworld import GridWorldMDP
from mdp.cats_v_monsters.cats_vs_monsters import CatsVMonstersMDP
from mdp.mdp_base import MDPBase

STATE_TUPLE_LEN = 2
MAX_GRID_CELLS = 100


# In[ ]:


class DiscreteMC(MDPBase):
    """Tabular approximation of MountainCar dynamics for planning."""

    ACTIONS = ("push_left", "coast", "push_right")

    def __init__(self, pos_bins: int = 41, vel_bins: int = 41):
        self.pos_min = -1.2
        self.pos_max = 0.6
        self.vel_min = -0.07
        self.vel_max = 0.07
        self.goal_pos = 0.5
        self.force_map = {
            "push_left": -1.0,
            "coast": 0.0,
            "push_right": 1.0,
        }

        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.pos_grid = np.linspace(self.pos_min, self.pos_max, pos_bins)
        self.vel_grid = np.linspace(self.vel_min, self.vel_max, vel_bins)
        self.state_space = [
            (pos_idx, vel_idx)
            for pos_idx in range(pos_bins)
            for vel_idx in range(vel_bins)
        ]
        self.start_state = (self._value_to_index(-0.5, self.pos_grid), self._value_to_index(0.0, self.vel_grid))
        self.terminal_states = [
            state for state in self.state_space if self._state_position(state) >= self.goal_pos
        ]

    def _value_to_index(self, value: float, grid: np.ndarray) -> int:
        value = float(np.clip(value, grid[0], grid[-1]))
        return int(np.abs(grid - value).argmin())

    def _state_position(self, state) -> float:
        pos_idx, _ = state
        return float(self.pos_grid[pos_idx])

    def _state_velocity(self, state) -> float:
        _, vel_idx = state
        return float(self.vel_grid[vel_idx])

    def _values_to_state(self, pos: float, vel: float):
        pos = float(np.clip(pos, self.pos_min, self.pos_max))
        vel = float(np.clip(vel, self.vel_min, self.vel_max))
        return (
            self._value_to_index(pos, self.pos_grid),
            self._value_to_index(vel, self.vel_grid),
        )

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.ACTIONS

    def get_terminal_states(self):
        return self.terminal_states

    def is_state_valid(self, state):
        return state in self.state_space

    def is_terminal(self, state):
        return self._state_position(state) >= self.goal_pos

    def reset(self):
        return self.start_state

    def reward_function(self, state, action=None, next_state=None):
        target_state = next_state if next_state is not None else state
        return 0.0 if self.is_terminal(target_state) else -1.0

    def get_next_transitions(self, state, action):
        if self.is_terminal(state):
            return [(state, 1.0, 0.0)]

        pos = self._state_position(state)
        vel = self._state_velocity(state)
        accel = self.force_map[action]

        new_vel = vel + 0.001 * accel - 0.0025 * np.cos(3 * pos)
        new_vel = float(np.clip(new_vel, self.vel_min, self.vel_max))
        new_pos = pos + new_vel
        if new_pos < self.pos_min:
            new_pos = self.pos_min
            new_vel = 0.0
        terminal = new_pos >= self.goal_pos
        new_state = self._values_to_state(new_pos, new_vel)
        reward = 0.0 if terminal else -1.0
        return [(new_state, 1.0, reward)]

    def step(self, state, action):
        next_state, _, reward = self.get_next_transitions(state, action)[0]
        terminal = self.is_terminal(next_state)
        return next_state, reward, terminal


# In[ ]:


@dataclass(slots=True)
class EnvSpec:
    name: str
    builder: callable
    description: str

ENV_REGISTRY = {
    'gridworld': EnvSpec(
        name='GridWorld 5x5',
        builder=lambda: GridWorldMDP(rows=5, cols=5),
        description='Stochastic 5x5 grid with terminal goal.'
    ),
    'cats_vs_monsters': EnvSpec(
        name='Cats vs Monsters',
        builder=lambda: CatsVMonstersMDP(rows=5, cols=5),
        description='Grid with tasty food, monsters, and obstacles.'
    ),
    'mountaincar': EnvSpec(
        name='Mountain Car',
        builder=lambda: DiscreteMC(pos_bins=41, vel_bins=41),
        description='41x41 discretization of the classic MountainCar physics.'
    ),
}

POLICY_DIR = Path('policies')
POLICY_DIR.mkdir(parents=True, exist_ok=True)

ARROW_MAP = {
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→',
}


# In[4]:


def _format_policy_grid(policy: dict):
    tuple_states = [
        state for state in policy.keys()
        if isinstance(state, tuple)
        and len(state) == STATE_TUPLE_LEN
        and all(isinstance(x, int) for x in state)
    ]
    if not tuple_states:
        return None
    max_r = max(r for r, _ in tuple_states)
    max_c = max(c for _, c in tuple_states)
    # Avoid dumping enormous grids (e.g., MountainCar 41x41 discretization)
    if (max_r + 1) * (max_c + 1) > MAX_GRID_CELLS:
        return None
    grid = [['·' for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for state in tuple_states:
        r, c = state
        action = policy.get(state)
        grid[r][c] = ARROW_MAP.get(action, action or '·')
    lines = ['Policy grid (rows top→bottom, cols left→right):']
    for row in grid:
        lines.append(' '.join(row))
    return '\n'.join(lines)


def save_optimal_policy(env_key, *, gamma=0.99, delta=1e-4):
    if env_key not in ENV_REGISTRY:
        raise KeyError(f'Unknown environment: {env_key}')
    spec = ENV_REGISTRY[env_key]
    mdp = spec.builder()
    solver = ValueIteration(mdp)
    history = solver.run_value_iteration(gamma=gamma, delta=delta)
    final_iter = max(history.keys())
    value_function, policy = history[final_iter]

    lines = [
        'Optimal policy via Value Iteration',
        f'Environment: {spec.name} ({env_key})',
        f'Iterations: {final_iter}',
        f'Gamma={gamma}, Delta={delta}',
        '',
        'state -> best action',
    ]
    for state in sorted(policy.keys()):
        lines.append(f'{state} -> {policy[state]}')

    grid = _format_policy_grid(policy)
    if grid:
        lines.extend(['', grid])

    out_path = POLICY_DIR / f'{env_key}_optimal_policy.txt'
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {
        'path': out_path,
        'iterations': final_iter,
        'value_function': value_function,
        'policy': policy,
    }



# In[5]:


save_optimal_policy('gridworld')
save_optimal_policy('cats_vs_monsters')
save_optimal_policy('mountaincar', gamma=0.99, delta=1e-3)

