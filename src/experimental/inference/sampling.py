from itertools import product

import numpy as np
from scipy.stats import truncnorm
from tqdm import trange

from ...utils.constants import ParamTuple, p_limits, gamma_limits, R_limits, StepSizeTuple, StateTransition
from ...utils.inference.likelihood import expert_trajectory_likelihood
from ...utils.make_environment import Environment
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


def exp_parameter_proposal_truncnorm(
    previous_sample: ParamTuple,
    step_sizes: StepSizeTuple,
) -> ParamTuple:
    
    '''
    Takes in a previous sample of parameters of type ParamTuple and a step size and returns a new sample where each sample
    is sampled from a truncated normal distribution with \mu = sample, \sigma = step_size
    '''

    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


    # Truncated normal distribution for p
    p_dist = get_truncated_normal(
        mean=previous_sample.p, sd=step_sizes.p, low=p_limits[0], upp=p_limits[1]
    )
    p = p_dist.rvs()

    # Truncated normal distribution for gamma
    gamma_dist = get_truncated_normal(
        mean=previous_sample.gamma,
        sd=step_sizes.gamma,
        low=gamma_limits[0],
        upp=gamma_limits[1],
    )
    gamma = gamma_dist.rvs()

    # Truncated normal distribution for R
    R_dist = get_truncated_normal(
        mean=previous_sample.R, sd=step_sizes.R, low=R_limits[0], upp=R_limits[1]
    )
    R = R_dist.rvs()

    return ParamTuple(p=p, gamma=gamma, R=R)


def exp_bayesian_parameter_learning(
    # TODO: Find an appropriate data structure for expert trajectories
    # It needs to account for the possibility of multiple trajectories per environment
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    sample_size: int,
    goal_states: np.array,
    n_states: int,
    current_sample: ParamTuple = None,
):
    # Samples from the posterior
    posterior_samples: list[ParamTuple] = []
    
    step_sizes = (0.1, 0.1, 0.1)

    # Start the chain at the previous sample if provided, otherwise sample from the prior
    if current_sample is None:
        current_sample = prior_sample(n_states)

        current_gamma = current_sample
        current_p = current_sample
        current_R = current_sample

    n_accepted_gamma = 0
    n_accepted_p = 0
    n_accepted_R = 0


    it = trange(sample_size, desc="Posterior sampling", leave=False)
    for k in it:

        # Create a new proposal for (p_i, gamma_i)
        proposed_parameter: ParamTuple = exp_parameter_proposal_truncnorm(
            current_sample, step_sizes=step_sizes
        )


        #now create 3 chains, one for each sample, e.g. each sample (gamma, p, R) is the proposed (gamma) and the previous (p,R)
        proposed_gamma = ParamTuple(p=current_sample.p, gamma=proposed_parameter.gamma, R=current_sample.R)
        proposed_p = ParamTuple(p=proposed_parameter.p, gamma=current_sample.gamma, R=current_sample.R)
        proposed_R = ParamTuple(p=current_sample.p, gamma=current_sample.gamma, R=proposed_parameter.R)



        def metropolis_hastings(proposed_sample, current_sample):
            
            likelihood_proposed = expert_trajectory_likelihood(
                proposed_sample, expert_trajectories, goal_states
            )

            likelihood_current = expert_trajectory_likelihood(
                current_sample, expert_trajectories, goal_states
            )            

            accepted = False
            # Check if we accept the proposal
            quotient = likelihood_proposed / likelihood_current
            if np.random.uniform(0, 1) < quotient:
                current_sample = proposed_sample
                accepted = True

            return current_sample, accepted
        

        
        current_gamma, gamma_accepted = metropolis_hastings(proposed_gamma, current_gamma)
        n_accepted_gamma += int(gamma_accepted)
        acc_rate_gamma = n_accepted_gamma/(k+1)

        current_p, p_accepted = metropolis_hastings(proposed_p, current_p)
        n_accepted_p += int(p_accepted)
        acc_rate_p = n_accepted_p/(k+1)

        current_R, R_accepted = metropolis_hastings(proposed_R, current_R)
        n_accepted_R += int(R_accepted)
        acc_rate_R = n_accepted_R/(k+1)

        current_sample = ParamTuple(p=current_p.p, gamma=current_gamma.gamma, R=current_R.R)

        posterior_samples.append(current_sample)


        def tune_stepsize_to_acc_rate(n_accepted, step_size):
        # Based on current acceptance rates, adjust step size and n_steps
            acceptance_rate = n_accepted / (k + 1)
            if acceptance_rate > 0.25:
                step_size = round(min(1, step_size + 0.01), 3)
            elif acceptance_rate < 0.21:
                step_size = round(max(0.01, step_size - 0.01), 3)

            return step_size
        
        _step_size_p = tune_stepsize_to_acc_rate(n_accepted=n_accepted_p, step_size=_step_size_p)
        _step_size_gamma = tune_stepsize_to_acc_rate(n_accepted=n_accepted_gamma, step_size=_step_size_gamma)
        _step_size_R = tune_stepsize_to_acc_rate(n_accepted=n_accepted_R, step_size=_step_size_R)

        step_sizes = StepSizeTuple(stepsize_p=_step_size_p, stepsize_gamma=_step_size_gamma, stepsize_R=_step_size_R)

        it.set_postfix(
            {
                "Acceptance rate gamma": round(100 * acc_rate_gamma, 1),
                "Acceptance rate p": round(100 * acc_rate_p, 1),
                "Acceptance rate R": round(100 * acc_rate_R, 1),
            }
        )

    return posterior_samples





'''
Legacy functions
'''



def get_parameter_sample(
    n_samples: int, N: int, M: int, ranges=[[0.5, 0.999], [0.5, 0.999], [1, 10]]
):
    """
    Returns a list of prior samples of (T_p, \gamma, R)

    Args:
    - n_samples, int, number of samples to generate
    - n_states, number of states of the maze, this is required for the reward samples as we generate a reward for each state
    - ranges, optional, specifies the ranges from which we sample for each argument, is of shape [[lower_range_gamma, higher_range_gamma
    ], [lower_range_p, higher_range_p], [lower_range_R, higher_range_R]]. Ranges for R must be integers and are divided by 10.
    """
    n_states = N*M

    n_cbrt = int(np.cbrt(n_samples))
    ps = np.linspace(ranges[0][0], ranges[0][1], n_cbrt)
    gammas = np.linspace(ranges[1][0], ranges[1][1], n_cbrt)
    Rs = np.random.randint(ranges[2][0], ranges[2][1], size=(n_cbrt, n_states)) / 10

    print("Update Rewards, create richer reward landscape")
    for R in Rs:
        R = np.reshape(R, (int(np.sqrt(n_states)), int(np.sqrt(n_states))))
        rand_num = np.random.random()

        if rand_num < 0.999:
            R[N-5, M-1] += 3
            R[N-4, M-1] += 3
            R[N-5, M-2] += 3
            R[N-4, M-2] += 3

            R[2, 2] += -2
            R[3, 3] += -2
            R[2, 3] += -2
            R[3, 2] += -2

            R[0, 0] += 0.5
            R[1, 1] += 0.5
            R[0, 1] += 0.5
            R[1, 0] += 0.5

        else:
            R[N-2, M-1] += 3
            R[N-1, M-1] += 3
            R[N-2, M-2] += 3
            R[N-1, M-2] += 3

            R[2, 2] += -2
            R[3, 3] += -2
            R[2, 3] += -2
            R[3, 2] += -2

            R[0, 0] += 0.5
            R[1, 1] += 0.5
            R[0, 1] += 0.5
            R[1, 0] += 0.5

        R = np.reshape(R, n_states)

    samples = list(product(ps, gammas, Rs))
    return samples
