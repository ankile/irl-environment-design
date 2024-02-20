from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..make_environment import Environment
from ..constants import ParamTuple, StateTransition
from ..inference.likelihood import expert_trajectory_log_likelihood




class PosteriorInference():


    '''
    Calculate Statistics for Posterior Distribution.

    Args:
    - resolution: int, mesh resolution of the posterior distribution.
    '''

    def __init__(self, 
                 expert_trajectories: List[tuple[Environment, List[StateTransition]]],
                 resolution: int=15) -> None:
        

        self.expert_trajectories = expert_trajectories
        self.resolution = resolution
        self.min_gamma = 0.05
        self.max_gamma = 0.95
        self.min_p = 0.05
        self.max_p = 0.95

        self.gammas = np.linspace(self.min_gamma, self.max_gamma, self.resolution)
        self.ps = np.linspace(self.min_p, self.max_p, self.resolution)


    
    def _validate_episode(self,
                          episode):
        
        assert f"episode={episode}" in self.posterior_distribution, f"Posterior Distribution for this episode does not exist yet. Only the following episodes exist: {self.posterior_distribution.keys()}"



    def calculate_posterior(self, 
                            num_episodes: int):
        
        '''
        Calculate the posterior distribution for episodes 1,..,num_episodes.

        Args:
        - episode: int, number of episodes for which the posterior should be evaluated.
        '''

        num_episodes_recorded = len(self.expert_trajectories)
        assert num_episodes <= num_episodes_recorded+1, f"episode is larger than number of available episodes. episode = {num_episodes}, number of played episodes = {num_episodes_recorded+1}"
        del num_episodes_recorded

        self.posterior_distribution: dict = {}

        
        for episode in range(num_episodes):

            '''
            Calculate Posterior Distribution for observations.
            '''

            #Arrays to loop over and store results.
            log_likelihoods: np.ndarray = np.zeros(shape = (self.resolution, self.resolution))


            #Observations up to current episode.
            expert_trajectories = self.expert_trajectories[:episode]


            #Calculate log-likelihood for each (p, gamma) sample.
            for idx_p, p in enumerate(self.ps):
                for idx_gamma, gamma in enumerate(self.gammas):

                    proposed_parameter = ParamTuple(p=p, gamma=gamma, R=None)

                    likelihood = expert_trajectory_log_likelihood(
                        proposed_parameter, expert_trajectories
                    )
                    log_likelihoods[idx_p, idx_gamma] = likelihood

            #Save log likelihoods.
            self.posterior_distribution[f"episode={episode}"] = log_likelihoods


    def plot(self,
             episode:int,
             param_values: ParamTuple=None,
             plot_mean: bool=True,
             plot_MAP: bool=True,
             show_true_prob: bool=True):

        '''
        Plot posterior distribution.

        Args:
        - episode: up to which episode the posterior is plotted.
        - param_values, ParamTuple: true parameter values.
        '''

        self._validate_episode(episode=episode)

        #Things to plot.
        posterior_dist = self.posterior_distribution[f"episode={episode}"]
        
        #Posterior Distribution.
        im = plt.imshow(
        posterior_dist, cmap="viridis",origin="lower")
        plt.colorbar(im, orientation="vertical")

        #Cosmetics.
        plt.xlabel("$\gamma$")
        plt.ylabel("p")
        plt.yticks(np.arange(self.resolution), np.round(self.ps, 2))
        plt.xticks(np.arange(self.resolution), np.round(self.gammas, 2), rotation='vertical')
        plt.title(f"Posterior over $p$ and $\gamma$\nlog likelihood after {episode} round(s).")


        #Show true values.
        if param_values:

            index_p_true = (np. abs(self.ps - param_values.p)). argmin()
            index_gamma_true = (np. abs(self.gammas - param_values.gamma)). argmin()


            #Show probability of true parameters.
            if show_true_prob:
                prob_true = self.prob_true(episode=episode, true_params=param_values)
                plt.plot(index_gamma_true, index_p_true, "D", label = f"True: (p,$\gamma$)={param_values.p,param_values.gamma}" + r", $\mathbb{P}$" + f"={round(prob_true,4)}")

            else:

                plt.plot(index_gamma_true, index_p_true, "D", label = f"True: (p,$\gamma$)=({param_values.p,param_values.gamma})")

            plt.legend()


        #Show posterior mean values.
        if plot_mean:
            mean_params = self.mean(episode=episode)

            index_p_mean = (np.abs(self.ps - mean_params.p)).argmin()
            index_gamma_mean = (np.abs(self.gammas - mean_params.gamma)).argmin()

            plt.plot(index_gamma_mean, index_p_mean, "D", label = f"Mean: (p,$\gamma$)={round(mean_params.p,2), round(mean_params.gamma,2)}")
            plt.legend()   


        #Show posterior MAP values.
        if plot_MAP:
            MAP_params = self.MAP(episode=episode)

            index_p_MAP = (np.abs(self.ps - MAP_params.p)).argmin()
            index_gamma_MAP = (np.abs(self.gammas - MAP_params.gamma)).argmin()

            plt.plot(index_gamma_MAP, index_p_MAP, "D", label = f"MAP: (p,$\gamma$)={round(MAP_params.p,2), round(MAP_params.gamma,2)}")
            plt.legend()



    def mean(self,
             episode:int):

        self._validate_episode(episode=episode)

        #Arrays to take mean over.
        posterior_probabilities = np.exp(self.posterior_distribution[f"episode={episode}"])

        self.total_probability = np.sum(posterior_probabilities, axis=(0,1))
        
        #Calculate mean.
        mean_p = np.sum(self.ps * np.sum(posterior_probabilities, axis=1))/self.total_probability
        mean_gamma = np.sum(self.gammas * np.sum(posterior_probabilities, axis=0))/self.total_probability


        return ParamTuple(p=mean_p, gamma=mean_gamma, R=None)
    

    def MAP(self,
            episode:int):
        
        '''
        Returns Maximum a Posteriori (MAP) of posterior distribution.
        '''
        
        self._validate_episode(episode=episode)

        #Get index where posterior distribution is largest.
        posterior_distribution = self.posterior_distribution[f"episode={episode}"]
        map_index = np.unravel_index(np.argmax(posterior_distribution, axis=None), posterior_distribution.shape)

        #Get corresponding p and gamma values.
        map_p = self.ps[map_index[0]]
        map_gamma = self.gammas[map_index[1]]

        return ParamTuple(p=map_p, gamma=map_gamma, R=None)
    

    def prob_true(self,
                  episode:int,
                  true_params: ParamTuple):
        
        '''
        Returns Probability of True Values according to posterior.
        '''

        self._validate_episode(episode=episode)


        index_p_true = (np.abs(self.ps - true_params.p)).argmin()
        index_gamma_true = (np.abs(self.gammas - true_params.gamma)).argmin()

        posterior_distribution = self.posterior_distribution[f"episode={episode}"]

        likelihood_true = posterior_distribution[index_p_true, index_gamma_true]
        probability_true = likelihood_true / np.sum(posterior_distribution, axis=(0,1))

        return probability_true