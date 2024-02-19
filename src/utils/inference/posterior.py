from typing import list

import numpy as np

from ..make_environment import Environment
from ..constants import ParamTuple, StateTransition
from ..inference.likelihood import expert_trajectory_log_likelihood




class PosteriorInference():


    '''
    Calculate Statistics for Posterior Distribution
    '''

    def __init__(self, 
                 expert_trajectories: list[tuple[Environment, list[StateTransition]]]) -> None:
        
        self.expert_trajectories = expert_trajectories
        self.min_gamma = 0.05
        self.max_gamma = 0.95
        self.min_p = 0.05
        self.max_p = 0.05

    def calculate_posterior(self, 
                            num_episodes: int,
                            resolution: int = 15):

        '''
        Calculate the posterior distribution for episodes 1,..,num_episodes.

        Args:
        - episode: int, number of episodes for which the posterior should be evaluated.
        - resolution: int, mesh resolution of the posterior distribution.
        '''

        num_episodes_recorded = len(self.expert_trajectories)
        assert num_episodes > num_episodes_recorded, f"episode is larger than number of available episodes.
        episode = f{num_episodes}, number of played episodes = f{num_episodes_recorded}"
        del num_episodes_recorded

        self.posterior_distribution: dict = {}

        
        for idx_episode, episode in enumerate(num_episodes):

            '''
            Calculate Posterior Distribution for observations.
            '''

            #Arrays to loop over and store results.
            gammas = np.linspace(0.5, 0.95, resolution)
            ps = np.linspace(0.95, 0.5, resolution)
            log_likelihoods: np.ndarray = np.zeros(shape = (resolution, resolution))


            #Observations up to current episode.
            expert_trajectories = self.expert_trajectories[:episode]

            #Calculate log-likelihood for each (p, gamma) sample.
            for idx_p, p in enumerate(ps):
                for idx_gamma, gamma in enumerate(gammas):

                    proposed_parameter = ParamTuple(p=p, gamma=gamma, R=None)

                    likelihood = expert_trajectory_log_likelihood(
                        proposed_parameter, expert_trajectories
                    )
                    log_likelihoods[idx_p, idx_gamma] = likelihood

            self.posterior_distribution[f"episode={idx_episode}"] = log_likelihoods
