import numpy as np
import torch

from ..constants import ParamTuple, StateTransition, beta_agent
from ..make_environment import Environment, transition_matrix, insert_walls_into_T
from ..optimization import soft_q_iteration

'''
Functions to calculate the log-likelihood of trajectories
'''

# @njit
def compute_log_likelihood(T, policy, trajectory):
    log_likelihood = 0.0
    for s, a, next_s in trajectory[:-1]:
        if (T[s, a, next_s] * policy[s, a]) == 0:
            print("encountered an impossible transition in my trajectory. this can't be. ignoring it for likelihood")
            print("current (s,a,next_s) tuple: ", (s,a,next_s))
        else:
            log_likelihood += np.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


def log_likelihood_torch(T, policy, trajectory):
    log_likelihood = torch.tensor(0.0)
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += torch.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


def expert_trajectory_log_likelihood(
    parameter_sample: ParamTuple,
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    goal_states: np.array
) -> float:
    
    '''
    Computes the log likelihood of an expert trajectory according to a parameter sample.
    Log-likelihood is averaged over length of trajectories.
    '''

    log_likelihood = 0.0

    for env, trajectories in expert_trajectories:
        T_agent = transition_matrix(env.N, env.M, p=parameter_sample.p, absorbing_states=goal_states)
        T_agent = insert_walls_into_T(T_agent, wall_indices=env.wall_states) #this is new
        policy = soft_q_iteration(
            parameter_sample.R, T_agent, gamma=parameter_sample.gamma, beta=beta_agent
        )
        for traj in trajectories:
            log_likelihood += compute_log_likelihood(env.T_true, policy, traj)

    if log_likelihood == -np.inf:
        print("log likelihood is negative infinity. sth is weird.")

    return log_likelihood


def expert_trajectory_likelihood(
    parameter_sample: ParamTuple,
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    goal_states: np.array
) -> float:
    
    '''
    Computes the likelihood of an expert trajectory according to a parameter sample.
    '''    
    
    return np.exp(expert_trajectory_log_likelihood(parameter_sample, expert_trajectories, goal_states))