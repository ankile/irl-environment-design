""" This concerns environment design for reward learning. """

from tqdm import trange
from gym_minigrid.wrappers import *
from auxiliary.auxiliary import *
from auxiliary.mdp_solver import *
from maze_env import ConstructedMazeEnv
from env_design import *
from multi_env_birl import *
from random import sample


size = 9
base_env = ConstructedMazeEnv(size=size)
n_environment_samples = 1_000
n_reward_samples = 750
uniform_samples = [np.random.uniform(0, 1, size**2) for _ in range(n_reward_samples)]
birl_sample_size = 1_000
observations = []
n_episodes = 3
obs_per_eps = 2
for episode in trange(n_episodes, desc="Episode"):
    if episode == 0:
        s_reward = uniform_samples
        plot_heatmaps(
            [0.5 for _ in range(size**2)],
            [1 for _ in range(size**2)],
            episode=episode,
        )
        # walls = []
        walls = domain_randomisation(base_env)  # domain randomisation
    else:
        (
            posterior_samples,
            posterior_mean,
            posterior_map,
            posterior_std,
        ) = bayesian_reward_learning(
            base_env, observations, birl_sample_size, proposal_distr="grid"
        )
        s_reward = sample(posterior_samples[-5000:], n_reward_samples)
        plot_heatmaps(posterior_mean, posterior_std, episode=episode)
        m_reward = sum(s_reward) / len(s_reward)
        # candidate_walls, bayes_regret = brute_force_maze_design(
        #     base_env, s_reward, m_reward, n_environment_samples
        # )
        candidate_walls, bayes_regret = brute_force_maze_design_likelihood(
            base_env, s_reward, m_reward, n_environment_samples, observations
        )
        walls = candidate_walls[np.argmax(bayes_regret)]
        # walls = []  # fixed maze
        # walls = domain_randomisation(base_env)  # domain randomisation
    env = ConstructedMazeEnv(size=size, walls=walls)
    save_initial_render(env, episode + 1)
    for _ in range(obs_per_eps):
        obs = get_expert_trajectory(env)
        observations.append(obs)
    env.close()
(
    posterior_samples,
    posterior_mean,
    posterior_map,
    posterior_std,
) = bayesian_reward_learning(
    base_env, observations, birl_sample_size, proposal_distr="grid"
)
plot_heatmaps(posterior_mean, posterior_std, episode=n_episodes)
