from itertools import product

import numpy as np
from scipy.stats import truncnorm
from tqdm import trange

from ..constants import ParamTuple, p_limits, gamma_limits, R_limits, StateTransition, KnownParameter, beta_agent
from .likelihood import expert_trajectory_log_likelihood
from ..make_environment import Environment, transition_matrix, insert_walls_into_T

'''
Functions for posterior sampling using Metropolis Hastings
'''



def prior_sample(n_states) -> ParamTuple:
    p = np.random.uniform(*p_limits)
    gamma = np.random.uniform(*gamma_limits)
    Rs = np.random.uniform(*R_limits, size=(n_states))

    return ParamTuple(p, gamma, Rs)


def parameter_proposal(previous_sample: ParamTuple, step_size: float, n_states: int) -> ParamTuple:
    p = np.random.normal(previous_sample.p, step_size)
    p = np.clip(p, *p_limits)

    gamma = np.random.normal(previous_sample.gamma, step_size)
    gamma = np.clip(gamma, *gamma_limits)

    reward_step = np.random.choice(
    [-step_size, 0, step_size], n_states, p=(0.15, 0.7, 0.15)
    )
    R = previous_sample.R + reward_step
    R = R.clip(min=0, max=1)

    return ParamTuple(p, gamma, R)


def parameter_proposal_truncnorm(
    previous_sample: ParamTuple,
    step_size: float,
) -> ParamTuple:
    
    '''
    Takes in a previous sample of parameters of type ParamTuple and a step size and returns a new sample where each sample
    is sampled from a truncated normal distribution with \mu = sample, \sigma = step_size
    '''

    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


    # Truncated normal distribution for p
    p_dist = get_truncated_normal(
        mean=previous_sample.p, sd=step_size, low=p_limits[0], upp=p_limits[1]
    )
    p = p_dist.rvs()

    # Truncated normal distribution for gamma
    gamma_dist = get_truncated_normal(
        mean=previous_sample.gamma,
        sd=step_size,
        low=gamma_limits[0],
        upp=gamma_limits[1],
    )
    gamma = gamma_dist.rvs()

    # Truncated normal distribution for R
    R_dist = get_truncated_normal(
        mean=previous_sample.R, sd=step_size, low=R_limits[0], upp=R_limits[1]
    )
    R = R_dist.rvs()

    return ParamTuple(p=p, gamma=gamma, R=R)


# def parameter_proposal_mala(previous_sample: ParamTuple,
#                             step_size,
#                             expert_trajectories,
#                             goal_states) -> ParamTuple:
    
#     grad = 0

#     for env, trajectories in expert_trajectories:
#         optimizer.zero_grad()
#         T_agent = transition_matrix(env.N, env.M, p=previous_sample.p, absorbing_states=goal_states)
#         T_agent = insert_walls_into_T(T_agent, wall_indices=env.wall_states) #this is new
#         policy = soft_q_iteration(
#             previous_sample.R, T_agent, gamma=previous_sample.gamma, beta=beta_agent
#         )

#     pass



def bayesian_parameter_learning(
    # TODO: Find an appropriate data structure for expert trajectories
    # It needs to account for the possibility of multiple trajectories per environment
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    sample_size: int,
    n_states: int = 0, #TODO, remove this argument.
    previous_sample: ParamTuple = None
):
    
    # Samples from the posterior
    posterior_samples: list[ParamTuple] = []
    n_accepted = 0
    step_size = 0.1

    # Start the chain at the previous sample if provided, otherwise sample from the prior
    if previous_sample is None:
        previous_sample = prior_sample(n_states)

    old_log_likelihood = expert_trajectory_log_likelihood(previous_sample, expert_trajectories)

    it = trange(sample_size, desc="Posterior sampling", leave=False)
    for k in it:

        # Create a new proposal for (p_i, gamma_i)
        proposed_parameter: ParamTuple = parameter_proposal_truncnorm(
            previous_sample, step_size=step_size
        )

        log_likelihood = expert_trajectory_log_likelihood(
            proposed_parameter, expert_trajectories
        )

        # Check if we accept the proposal
        p = log_likelihood  # We don't multiply by the prior because it's uniform
        p_old = old_log_likelihood
        quotient = np.exp(p-p_old)
        # quotient = np.exp(p - p_old)
        if np.random.uniform(0, 1) < quotient:
            previous_sample = proposed_parameter
            old_log_likelihood = log_likelihood
            n_accepted += 1
        posterior_samples.append(previous_sample)

        # # Based on current acceptance rates, adjust step size and n_steps
        acceptance_rate = n_accepted / (k + 1)
        # if acceptance_rate > 0.25:
        #     step_size = round(min(1, step_size + 0.01), 3)
        # elif acceptance_rate < 0.21:
        #     step_size = round(max(0.01, step_size - 0.01), 3)

        it.set_postfix(
            {
                "Acceptance rate": round(100 * acceptance_rate, 1),
                "step_size": step_size,
            }
        )

    return posterior_samples





# '''
# Legacy functions
# '''



# def get_parameter_sample(
#     n_samples: int, N: int, M: int, ranges=[[0.5, 0.999], [0.5, 0.999], [1, 10]]
# ):
#     """
#     Returns a list of prior samples of (T_p, \gamma, R)

#     Args:
#     - n_samples, int, number of samples to generate
#     - n_states, number of states of the maze, this is required for the reward samples as we generate a reward for each state
#     - ranges, optional, specifies the ranges from which we sample for each argument, is of shape [[lower_range_gamma, higher_range_gamma
#     ], [lower_range_p, higher_range_p], [lower_range_R, higher_range_R]]. Ranges for R must be integers and are divided by 10.
#     """
#     n_states = N*M

#     n_cbrt = int(np.cbrt(n_samples))
#     ps = np.linspace(ranges[0][0], ranges[0][1], n_cbrt)
#     gammas = np.linspace(ranges[1][0], ranges[1][1], n_cbrt)
#     Rs = np.random.randint(ranges[2][0], ranges[2][1], size=(n_cbrt, n_states)) / 10

#     print("Update Rewards, create richer reward landscape")
#     for R in Rs:
#         R = np.reshape(R, (int(np.sqrt(n_states)), int(np.sqrt(n_states))))
#         rand_num = np.random.random()

#         if rand_num < 0.999:
#             R[N-5, M-1] += 3
#             R[N-4, M-1] += 3
#             R[N-5, M-2] += 3
#             R[N-4, M-2] += 3

#             R[2, 2] += -2
#             R[3, 3] += -2
#             R[2, 3] += -2
#             R[3, 2] += -2

#             R[0, 0] += 0.5
#             R[1, 1] += 0.5
#             R[0, 1] += 0.5
#             R[1, 0] += 0.5

#         else:
#             R[N-2, M-1] += 3
#             R[N-1, M-1] += 3
#             R[N-2, M-2] += 3
#             R[N-1, M-2] += 3

#             R[2, 2] += -2
#             R[3, 3] += -2
#             R[2, 3] += -2
#             R[3, 2] += -2

#             R[0, 0] += 0.5
#             R[1, 1] += 0.5
#             R[0, 1] += 0.5
#             R[1, 0] += 0.5

#         R = np.reshape(R, n_states)

#     samples = list(product(ps, gammas, Rs))
#     return samples
