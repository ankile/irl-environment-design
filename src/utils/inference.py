from itertools import product

import numpy as np
import torch

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
        chosen_action = np.random.choice(n_actions, p=action_probabilities)

        # Manually sample next_state based on T_true
        next_state = np.random.choice(
            n_states, p=T_true[current_state, chosen_action])

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


# Make function to calculate log-likelihood of a trajectory given a transition matrix and policy
# @njit
def compute_log_likelihood(T, policy, trajectory):
    log_likelihood = 0.0
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += np.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


def log_likelihood_torch(T, policy, trajectory):
    log_likelihood = torch.tensor(0.0)
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += torch.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


# This function is a standin for the BIRL procedure that will create a proper posterior sampling


def get_parameter_sample(
    n_samples: int, n_states: int, ranges=[[0.5, 0.999], [0.5, 0.999], [1, 10]]
):
    """
    Returns a list of prior samples of (T_p, \gamma, R)

    Args:
    - n_samples, int, number of samples to generate
    - n_states, number of states of the maze, this is required for the reward samples as we generate a reward for each state
    - ranges, optional, specifies the ranges from which we sample for each argument, is of shape [[lower_range_gamma, higher_range_gamma
    ], [lower_range_p, higher_range_p], [lower_range_R, higher_range_R]]. Ranges for R must be integers and are divided by 10.
    """

    n_cbrt = int(np.cbrt(n_samples))
    ps = np.linspace(ranges[0][0], ranges[0][1], n_cbrt)
    gammas = np.linspace(ranges[1][0], ranges[1][1], n_cbrt)
    Rs = np.random.randint(ranges[2][0], ranges[2][1], size=(n_cbrt, n_states)) / 10

    # if n_states == 36:
    #     print("Update Rewards, create richer reward landscape")
    #     for R in Rs:
    #         R = np.reshape(R, (int(np.sqrt(n_states)), int(np.sqrt(n_states))))
    #         rand_num = np.random.random()

    #         if rand_num < 0.5:
    #             R[4, 4] += 3
    #             R[5, 5] += 3
    #             R[4, 5] += 3
    #             R[5, 4] += 3

    #             R[2, 2] += -2
    #             R[3, 3] += -2
    #             R[2, 3] += -2
    #             R[3, 2] += -2

    #             R[0, 0] += 0.5
    #             R[1, 1] += 0.5
    #             R[0, 1] += 0.5
    #             R[1, 0] += 0.5

    #         else:
    #             R[1, 4] += 3
    #             R[2, 5] += 3
    #             R[1, 5] += 3
    #             R[2, 4] += 3

    #             R[2, 2] += -2
    #             R[3, 3] += -2
    #             R[2, 3] += -2
    #             R[3, 2] += -2

    #             R[0, 0] += 0.5
    #             R[1, 1] += 0.5
    #             R[0, 1] += 0.5
    #             R[1, 0] += 0.5

    #         R = np.reshape(R, n_states)

    samples = list(product(ps, gammas, Rs))
    return samples