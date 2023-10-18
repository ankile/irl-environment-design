""" Implementation of a Sample-Based Bayesian Multi-Environment Inverse Reinforcement Learning Algorithm """

import numpy as np
import scipy
from tqdm import trange
from auxiliary.mdp_solver import *


# We work on the reward simplex. The reward function can still be scaled later.


# observations is a sequence of pairs [env, traj], where env is a gym object and traj is a sequence of state-action pairs.
def bayesian_reward_learning(
    base_env,
    observations,
    sample_size,
    proposal_distr="grid",
    last_reward=None,
    beta=20,
):
    if proposal_distr == "grid":
        beta = 20
    elif proposal_distr == "simplex":
        beta = 40
    # summarise multiple trajectories in the same environment
    compact_obs = make_obs_compact(observations)

    # samples from the posterior
    posterior_samples = []
    # count no of accepted proposals
    n_accepted = 0
    # dynamic step size
    step_size = 0.1

    # dynamic n_steps
    n_steps = 100
    if last_reward is None:
        if proposal_distr == "grid":
            last_reward = grid_prior(base_env)
        elif proposal_distr == "simplex":
            last_reward = simplex_prior(base_env)
    old_likelihood, old_scales = get_likelihood(last_reward, compact_obs, beta=beta)
    it = trange(sample_size, desc="Sample", leave=False)
    for k in it:
        if proposal_distr == "grid":
            proposed_reward, pdf_proposal = grid_proposal(
                base_env, last_reward, step_size=step_size
            )
        elif proposal_distr == "simplex":
            proposed_reward, pdf_proposal = simplex_proposal(
                last_reward, n_steps=n_steps
            )

        likelihood, n_scales = get_likelihood(proposed_reward, compact_obs, beta=beta)
        if old_scales != n_scales:
            scale = (old_scales - n_scales) * 1e20
        else:
            scale = 1
        p = likelihood * scale
        p_old = old_likelihood
        quotient = p / p_old
        if np.random.uniform(0, 1) < quotient:
            last_reward = proposed_reward
            old_likelihood = likelihood
            old_scales = n_scales
            n_accepted += 1
        posterior_samples.append(last_reward)
        acceptance_rate = n_accepted / (k + 1)
        if acceptance_rate > 0.25:
            step_size = round(min(1, step_size + 0.01), 3)
            n_steps += 5
            n_steps = min(1000, n_steps)
        elif acceptance_rate < 0.21:
            step_size = round(max(0.01, step_size - 0.01), 3)
            n_steps -= 5
            n_steps = max(n_steps, 1)
        if k % 100 == 0:
            it.set_postfix(
                {
                    "Acceptance rate": round(100 * acceptance_rate, 1),
                    "step_size": step_size,
                    "n_steps": n_steps,
                }
            )
    posterior_mean = sum(posterior_samples[int(sample_size / 4) :]) / (
        sample_size - int(sample_size / 4)
    )
    posterior_std = np.std(np.array(posterior_samples[int(sample_size / 4) :]), axis=0)
    # get MAP
    mtx = np.matrix(posterior_samples)
    values, counts = np.unique(mtx, return_counts=True, axis=0)
    posterior_map = values[counts == np.max(counts)][0]
    # print("MAP No. Times", np.max(counts))
    return posterior_samples, posterior_mean, posterior_map, posterior_std


def make_obs_compact(observations):
    environments = [item[0] for item in observations]
    unique_env = list(set(environments))
    compact_obs = [[item] for item in unique_env]
    for obs in observations:
        for env_idx, env in enumerate(unique_env):
            if obs[0] is env:
                compact_obs[env_idx] += [obs[1]]
    return compact_obs


# get likelihood of reward under observations
def get_likelihood(reward, compact_obs, beta=20):
    likelihood = 1
    n_scales = 0
    for obs in compact_obs:
        obs[0].rewards = reward
        V, Q, pol = value_iteration(obs[0])
        Q_exp = np.exp(beta * Q)
        for traj in obs[1:]:
            for s, a in traj:
                likelihood *= Q_exp[s, a] / np.sum(Q_exp[s, :])
                # print(Q_exp[s, a] / np.sum(Q_exp[s, :]))
                # there can be issues with floating points, as likelihoods can be very small
                # we resolve the issue by scaling (and remembering how much we scaled)
                while likelihood < 1e-100:
                    likelihood *= 1e20
                    n_scales += 1
    return likelihood, n_scales


def get_log_likelihood(reward, compact_obs, beta=20):
    log_likelihood = 0
    for obs in compact_obs:
        obs[0].rewards = reward
        V, Q, pol = value_iteration(obs[0])
        Q_exp = np.exp(beta * Q)
        for traj in obs[1:]:
            for s, a in traj:
                log_term = beta * Q[s, a] - np.log(np.sum(Q_exp[s, :]))
                log_likelihood += log_term
    return log_likelihood


# Some priors and proposal distributions
def simplex_prior(env):
    reward = np.zeros(env.state_space.n)
    step_size = 0.01
    choices = np.random.choice(env.state_space.n, int(1 / step_size))
    for i in range(len(choices)):
        reward[choices[i]] += step_size
    return reward


def pdf_simplex_prior():
    return 1


def simplex_proposal(last_reward, n_steps=100):
    proposed_reward = last_reward.copy()
    step_size = 0.01
    for _ in range(n_steps):
        minus_dir = np.random.choice(np.where(proposed_reward > 0 + 1e-5)[0])
        proposed_reward[minus_dir] -= step_size
        plus_dir = np.random.choice(np.where(proposed_reward < 1 - step_size + 1e-5)[0])
        proposed_reward[plus_dir] += step_size
    proposed_reward = np.round(proposed_reward, 3)
    assert sum(proposed_reward) < 1 + 1e-5
    # print(sum(proposed_reward, "Imprecise proposal due to python floats!"))
    pdf_proposal = 1
    return proposed_reward, pdf_proposal


def grid_prior(env):
    return np.random.randint(1, 10, size=env.state_space.n) / 10


def grid_proposal(env, last_reward, step_size):
    proposed_reward = last_reward.copy()
    step = np.random.choice(
        [-step_size, 0, step_size], env.state_space.n, p=(0.15, 0.7, 0.15)
    )
    proposed_reward += step
    # print(step[0:10])
    proposed_reward = proposed_reward.clip(min=0, max=1)
    pdf_proposal = 1
    return proposed_reward, pdf_proposal


def pdf_grid_prior():
    return 1


# todo: make something like the below which functions well, maybe via the amount of states we change (and not just the magnitude)
def single_step_proposal(env, last_reward, step_size):
    proposed_reward = last_reward.copy()
    proposed_reward = np.random.choice([0, 1], env.state_space.n)
    # step_idx = np.random.choice(range(env.state_space.n))
    # if np.random.uniform(0, 1) < 0.5:
    #     proposed_reward[step_idx] += step_size
    # else:
    #     proposed_reward[step_idx] -= step_size
    pdf_proposal = 1
    return proposed_reward, pdf_proposal
