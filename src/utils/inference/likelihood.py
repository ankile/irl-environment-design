import numpy as np

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
            #this can happen if samples generate a uniform policy (e.g. due to low gamma and p) and the agent thereby takes
            #a step off the grid
            # print("encountered an impossible transition in my trajectory. this can't be. ignoring it for likelihood")
            # print("current (s,a,next_s) tuple: ", (s,a,next_s))
            # print("Transition probability: T[s, a, next_s]", T[s, a, next_s])
            # print("Policy probability: policy[s, a]", policy[s, a])
            pass
        else:
            log_likelihood += np.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


def expert_trajectory_log_likelihood(
    # parameter_sample: ParamTuple,
    transition_function,
    reward_function,
    gamma,
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    # goal_states: np.array
) -> float:
    
    '''
    Computes the log likelihood of an expert trajectory according to a parameter sample.
    Log-likelihood is averaged over length of trajectories.
    '''

    log_likelihood = 0.0
    # print("Parameter sample: ", parameter_sample)

    #Initialize Q, V, policy
    Q = None
    V = None
    policy = None

    for env, trajectories in expert_trajectories: #TODO do we need env here?
        # assert env.goal_states is not None, "Add goal states to environment."
        # T_agent = transition_matrix(env.N, env.M, p=parameter_sample.p, absorbing_states=env.goal_states)
        # T_agent = insert_walls_into_T(T_agent, wall_indices=env.wall_states) #this is new
        policy, Q, V = soft_q_iteration(
            reward_function, transition_function, gamma=gamma, beta=beta_agent, return_what="all", Q_init=Q, V_init=V, policy_init=policy
        )
        for traj in trajectories:
            len_traj = len(traj)
            log_likelihood += compute_log_likelihood(transition_function, policy, traj)/len_traj #TODO: which T here? env.T_true or T_agent?
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