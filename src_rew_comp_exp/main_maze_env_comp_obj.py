""" This concerns environment design for reward learning. """

from gym_minigrid.wrappers import *
from auxiliary.auxiliary import *
from auxiliary.mdp_solver import *
from maze_env import ConstructedMazeEnv
from env_design import *
from multi_env_birl import *
import time
from random import sample


# Experiment parameters
size = 9
base_env = ConstructedMazeEnv(size=size)
n_environment_samples = 16
n_reward_samples = 750
uniform_samples = [np.random.uniform(0, 1, size**2) for _ in range(n_reward_samples)]
birl_sample_size = 1_000
n_episodes = 3
obs_per_eps = 2


# Storage for observations
value_obs = []
likelihood_obs = []

regret_funcs = {
    "value": evaluate_value_regret_of_maze,
    "likelihood": evaluate_likelihood_regret_of_maze,
}


def bayesian_inf_and_env_search(
    base_env,
    n_reward_samples,
    birl_sample_size,
    value_obs,
    episode,
    candidate_walls,
    regret_type,
):
    (
        posterior_samples,
        posterior_mean,
        posterior_map,
        posterior_std,
    ) = bayesian_reward_learning(
        base_env, value_obs, birl_sample_size, proposal_distr="grid"
    )
    s_reward = sample(posterior_samples[-5000:], n_reward_samples)
    plot_heatmaps(
        posterior_mean, posterior_std, episode=episode, regret_type=regret_type
    )
    m_reward = sum(s_reward) / len(s_reward)

    bayes_regret = evaluate_regret_for_candidates(
        base_env,
        s_reward,
        m_reward,
        candidate_walls,
        regret_func=regret_funcs[regret_type],
    )

    arg_max = np.argmax(bayes_regret)
    print(f"Max {regret_type} regret: {bayes_regret[arg_max]} for walls {arg_max}")
    walls = candidate_walls[arg_max]
    return walls


for episode in range(n_episodes):
    if episode == 0:
        s_reward = uniform_samples
        plot_heatmaps(
            [0.5 for _ in range(size**2)],
            [1 for _ in range(size**2)],
            episode=episode,
        )
        value_walls = []
        likelihood_walls = []
    else:
        # Start by defining the set of candidate walls
        # so that it'll be the same set for each objective
        candidate_walls = get_environment_candidates(base_env, n_environment_samples)

        # First run bayesian reward learning for the walls from value objective
        value_walls = bayesian_inf_and_env_search(
            base_env,
            n_reward_samples,
            birl_sample_size,
            value_obs,
            episode,
            candidate_walls,
            "value",
        )

        # Then run bayesian reward learning for the walls from likelihood objective
        likelihood_walls = bayesian_inf_and_env_search(
            base_env,
            n_reward_samples,
            birl_sample_size,
            likelihood_obs,
            episode,
            candidate_walls,
            "likelihood",
        )

    value_env = ConstructedMazeEnv(size=size, walls=value_walls)
    likelihood_env = ConstructedMazeEnv(size=size, walls=likelihood_walls)

    save_initial_render(value_env, episode + 1, regret_type="value")
    save_initial_render(likelihood_env, episode + 1, regret_type="likelihood")

    for _ in range(obs_per_eps):
        obs = get_expert_trajectory(value_env)
        value_obs.append(obs)
        obs = get_expert_trajectory(likelihood_env)
        likelihood_obs.append(obs)

    value_env.close()
    likelihood_env.close()

(
    posterior_samples,
    posterior_mean,
    posterior_map,
    posterior_std,
) = bayesian_reward_learning(
    base_env, value_obs, birl_sample_size, proposal_distr="grid"
)
plot_heatmaps(posterior_mean, posterior_std, episode=n_episodes, regret_type="value")

(
    posterior_samples,
    posterior_mean,
    posterior_map,
    posterior_std,
) = bayesian_reward_learning(
    base_env, likelihood_obs, birl_sample_size, proposal_distr="grid"
)
plot_heatmaps(
    posterior_mean, posterior_std, episode=n_episodes, regret_type="likelihood"
)
