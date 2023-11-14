import numpy as np
import torch

from ..constants import ParamTuple, StateTransition
from ..make_environment import Environment, transition_matrix
from ..optimization import soft_q_iteration

'''
Functions to calculate the log-likelihood of trajectories
'''

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


def expert_trajectory_likelihood(
    parameter_sample: ParamTuple,
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    goal_states: np.array
) -> float:
    log_likelihood = 0.0

    for env, trajectories in expert_trajectories:
        T_agent = transition_matrix(env.N, env.M, p=parameter_sample.p, absorbing_states = goal_states)
        policy = soft_q_iteration(
            parameter_sample.R, T_agent, gamma=parameter_sample.gamma, beta=20.0
        )
        for traj in trajectories:
            log_likelihood += compute_log_likelihood(env.T_true, policy, traj)

    return np.exp(log_likelihood)