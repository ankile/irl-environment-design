import numpy as np


'''
Functions for generating trajectories according to a policy in a given environment
'''

def generate_trajectory(T_true, policy, absorbing_states, start_state=0, max_steps=100):
    trajectory = []
    current_state = start_state
    n_states, n_actions = policy.shape

    while len(trajectory) < max_steps:
        if current_state in absorbing_states:
            # Append the absorbing state
            trajectory.append((current_state, -1, -1))
            break
        # Sample an action based on the policy probabilities for the current state
        action_probabilities = policy[current_state]
        try:
            chosen_action = np.random.choice(n_actions, p=action_probabilities)
        except:
            print("action probabilities: ", action_probabilities)
            print("sum action probabilities: ", sum(action_probabilities))
        # Manually sample next_state based on T_true
        try:
            next_state = np.random.choice(
                n_states, p=T_true[current_state, chosen_action])
        except:
            print("action probabilities: ", T_true[current_state, chosen_action])
            print("sum action probabilities: ", sum(T_true[current_state, chosen_action]))            
        trajectory.append((current_state, chosen_action, next_state))
        current_state = next_state

    return np.array(trajectory)

def generate_n_trajectories(
    T_true, policy, absorbing_states, start_state=0, n_trajectories=100, max_steps=100
):
    trajectories = list()
    for _ in range(n_trajectories):
        trajectories.append(
            generate_trajectory(
                T_true,
                policy,
                absorbing_states,
                start_state=start_state,
                max_steps=max_steps,
            )
        )
    return trajectories