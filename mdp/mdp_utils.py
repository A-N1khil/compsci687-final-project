import numpy as np

def display_value_function(mdp, value_function):
    # Create a grid representation of the value function
    value_function_grid = np.full((mdp.rows, mdp.cols), None)
    for state, value in value_function.items():
        row, col = state
        value_function_grid[row, col] = value

    # Display the value function grid
    for row in value_function_grid:
        for value in row:
            if value is None:
                print("0.0000", end="\t")
            else:
                print(f"{value:.4f}", end="\t")
        print()


def display_policy(mdp, policy):
    # Define arrows as latex symbols using r
    action_symbols = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
        None: "F",
    }
    policy_grid = np.full((mdp.rows, mdp.cols), "X")
    for state, action in policy.items():
        row, col = state
        policy_grid[row, col] = action_symbols[action]

    policy_grid[mdp.food] = "G"

    # Display the policy grid
    for row in policy_grid:
        for symbol in row:
            print(symbol, end="\t")
        print()

def max_norm_error(v1: dict, v2: dict):
    """Compute the MaxNorm error between two value functions."""
    max_error = float("-inf")
    for state in list(v1.keys()):
        v1_state_value = v1[state] or 0.0
        v2_state_value = v2[state] or 0.0
        error = abs(v1_state_value - v2_state_value)
        if error > max_error:
            max_error = error
    return max_error
