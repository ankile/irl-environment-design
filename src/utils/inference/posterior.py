from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..make_environment import Environment
from ..constants import ParamTuple, StateTransition
from ..inference.likelihood import expert_trajectory_log_likelihood




class PosteriorInference():


    '''
    Calculate Statistics for Posterior Distribution.
    '''

    def __init__(self, 
                 expert_trajectories: List[tuple[Environment, List[StateTransition]]]) -> None:
        

        self.expert_trajectories = expert_trajectories
        self.min_gamma = 0.05
        self.max_gamma = 0.95
        self.min_p = 0.05
        self.max_p = 0.95


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
        assert num_episodes <= num_episodes_recorded+1, f"episode is larger than number of available episodes. episode = {num_episodes}, number of played episodes = {num_episodes_recorded+1}"
        del num_episodes_recorded

        self.posterior_distribution: dict = {}
        self.resolution = resolution

        
        for episode in range(num_episodes+1):

            '''
            Calculate Posterior Distribution for observations.
            '''

            #Arrays to loop over and store results.
            gammas = np.linspace(self.min_gamma, self.max_gamma, self.resolution)
            ps = np.linspace(self.min_p, self.max_p, self.resolution)
            log_likelihoods: np.ndarray = np.zeros(shape = (self.resolution, self.resolution))


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

            #Save log likelihoods.
            self.posterior_distribution[f"episode={episode}"] = log_likelihoods


    def plot(self,
             episode:int,
             param_values: ParamTuple=None):

        '''
        Plot posterior distribution.

        Args:
        - episode: up to which episode the posterior is plotted.
        - param_values, ParamTuple: true parameter values.
        '''

        assert f"episode={episode}" in self.posterior_distribution, f"Posterior Distribution for this episode does not exist yet. Only the following episodes exist: {self.posterior_distribution.keys()}"


        #Things to plot.
        posterior_dist = self.posterior_distribution[f"episode={episode}"]
        gammas_ticks = np.linspace(self.min_gamma, self.max_gamma, self.resolution)
        ps_ticks = np.linspace(self.min_p, self.max_p, self.resolution)
        
        
        #Posterior Distribution.
        im = plt.imshow(
        posterior_dist, cmap="viridis",origin="lower")
        plt.colorbar(im, orientation="vertical")

        #Cosmetics.
        plt.xlabel("$\gamma$")
        plt.ylabel("p")
        plt.yticks(np.arange(self.resolution), np.round(ps_ticks, 2))
        plt.xticks(np.arange(self.resolution), np.round(gammas_ticks, 2), rotation='vertical')
        plt.title(f"Posterior over $p$ and $\gamma$\nlog likelihood after {episode} round(s).")

        if param_values:
            index_p_true = (np. abs(ps_ticks - param_values.p)). argmin()
            index_gamma_true = (np. abs(gammas_ticks - param_values.gamma)). argmin()

            plt.plot(index_gamma_true, index_p_true, "og", label = "True Values")
            plt.legend()
