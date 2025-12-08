from mdp.mdp_base import MDPBase


class ValueIteration:
    def __init__(self, mdp: MDPBase):
        self.mdp = mdp

    def run_value_iteration(self, gamma=0.925, delta=1e-4, verbose=False):
        # Keep a count of iterations to see at which iteration we converge
        iteration = 0
        run_history = {}
        value_function = {state: 0.0 for state in self.mdp.get_state_space()}
        valid_states = [
            state for state in self.mdp.get_state_space() if self.mdp.is_state_valid(state)]

        while True:
            value_diff = 0
            new_value_function = value_function.copy()

            # Iterate over all states in the MDP
            for state in valid_states:
                if self.mdp.is_terminal(state):
                    continue

                # Computing action values for all possible actions from this state
                action_values = []
                for action in self.mdp.get_action_space():
                    action_val = 0.0
                    transitions = self.mdp.get_next_transitions(state, action)

                    for next_state, prob, reward in transitions:
                        action_val += prob * (
                            reward + gamma * value_function[next_state]
                        )

                    action_values.append(action_val)

                # Performing the Bellman update
                best_action_value = max(action_values)
                new_value_function[state] = best_action_value

                # Calculate the difference for convergence check
                value_diff = max(
                    value_diff, abs(best_action_value - value_function[state])
                )

            value_function = new_value_function
            iteration += 1
            policy = self._create_policy_from_value_function(value_function, gamma)
            run_history[iteration] = (value_function.copy(), policy.copy())
            if verbose:
                print(f"Iteration {iteration}, Value Difference: {value_diff:.6f}")

            # Checking convergence
            if value_diff < delta:
                break

        return run_history

    def _create_policy_from_value_function(self, value_function, gamma):
        policy = {state: None for state in self.mdp.get_state_space()}
        valid_states = [state for state in self.mdp.get_state_space() if self.mdp.is_state_valid(state)]
        for state in valid_states:
            if self.mdp.is_terminal(state):
                policy[state] = None
                continue

            # Determine the best action based on the current value function
            best_action = None
            best_action_value = float("-inf")
            for action in self.mdp.get_action_space():
                action_val = 0.0
                transitions = self.mdp.get_next_transitions(state, action)

                for next_state, prob, reward in transitions:
                    action_val += prob * (reward + gamma * value_function[next_state])

                if action_val > best_action_value:
                    best_action_value = action_val
                    best_action = action

            policy[state] = best_action

        return policy
