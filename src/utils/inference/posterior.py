from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..make_environment import Environment
from ..constants import ParamTuple, StateTransition
from ..inference.likelihood import expert_trajectory_log_likelihood




class   PosteriorInference():


    '''
    Calculate Statistics for Posterior Distribution.

    Args:
    - resolution: int, mesh resolution of the posterior distribution.
    '''

    def __init__(self,
                 base_environment: Environment,
                 expert_trajectories: List[tuple[Environment, List[StateTransition]]],
                 learn_what: list,
                 parameter_ranges: np.ndarray,
                 parameter_mesh: list,
                 parameter_mesh_shape: np.ndarray,
                 region_of_interest = None) -> None:
        

        self.base_environment = base_environment
        self.expert_trajectories = expert_trajectories
        # self.resolution = resolution
        self.learn_what = learn_what
        self.parameter_mesh = parameter_mesh
        self.parameter_mesh_shape = parameter_mesh_shape
        self.parameter_ranges = parameter_ranges
        self.region_of_interest = region_of_interest



    
    def _validate_episode(self,
                          episode):
        
        assert f"episode={episode}" in self.posterior_distribution, f"Posterior Distribution for this episode does not exist yet. Only the following episodes exist: {self.posterior_distribution.keys()}"



    def calculate_posterior(self, 
                            num_episodes = None,
                            episode = None):
        
        '''
        Calculate the posterior distribution for episodes 1,..,num_episodes or only for episode *episode*.

        Args:
        - episode: int, number of episodes for which the posterior should be evaluated.
        '''

        num_episodes_recorded = len(self.expert_trajectories)
        if num_episodes:
            assert num_episodes <= num_episodes_recorded, f"episode is larger than number of available episodes. episode = {num_episodes}, number of played episodes = {num_episodes_recorded}"
        del num_episodes_recorded
        assert (type(num_episodes) == int) or (num_episodes is None)
        assert (type(episode) == int) or (episode is None)

        def _compute_likelihood_for_episode(episode):
                
                #Arrays to loop over and store results.
                log_likelihoods: np.ndarray = np.zeros_like(self.parameter_mesh_shape.flatten())
                print(f"Beginning calculation of log-likelihood. Calculating {len(log_likelihoods)} samples.")


                #Observations up to current episode.
                expert_trajectories = self.expert_trajectories[:episode]



                #Calculate log-likelihood for each parameter sample.
                for idx_parameter, parameter in enumerate(self.parameter_mesh):

                        #If a ROI is given, only compute likelihoods within Region of Interest to save compute.
                        if self.region_of_interest is not None:
                            if idx_parameter not in self.region_of_interest:
                                log_likelihoods[idx_parameter] = -np.inf
                                continue


                        #Insert parameter values into custom functions.
                        _transition_func = self.base_environment.transition_function(*parameter.T)
                        _reward_func = self.base_environment.reward_function(*parameter.R)
                        _gamma = self.base_environment.gamma(*parameter.gamma)


                        #Calculate log-likelihood.
                        likelihood = expert_trajectory_log_likelihood(
                            transition_function=_transition_func,
                            reward_function=_reward_func,
                            gamma=_gamma,   
                            expert_trajectories=expert_trajectories
                        )

                        log_likelihoods[idx_parameter] = likelihood

                #Unflatten likelihoods.        
                log_likelihoods = np.array(log_likelihoods)
                log_likelihoods = np.reshape(log_likelihoods, self.parameter_mesh_shape.shape) #TODO, is this reshape correct? E.g. do we get the right axes?s
                
                del _transition_func, _reward_func, _gamma


                return log_likelihoods

        self.posterior_distribution: dict = {}

        if episode is not None:

            return _compute_likelihood_for_episode(episode)
        
        if num_episodes is not None:
            for episode in range(num_episodes+1):

                '''
                Calculate Posterior Distribution for observations.
                '''

                if episode == 0:
                    print(f"Calculate distribution of episode {episode}, e.g. the prior distribution.")
                else:
                    print(f"Calculate distribution of episode {episode}.")

                
                log_likelihoods = _compute_likelihood_for_episode(episode)
                #Save log likelihoods.
                self.posterior_distribution[f"episode={episode}"] = log_likelihoods





    def plot_posterior(self,
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

        #Normalize to get probabilities.
        posterior_dist = np.exp(posterior_dist)
        posterior_dist = posterior_dist/np.sum(posterior_dist)
        
        #Posterior Distribution.
        im = plt.imshow(
        posterior_dist, cmap="viridis",origin="upper")
        plt.colorbar(im, orientation="vertical")

        #Cosmetics.
        plt.xlabel("$\gamma$")
        plt.ylabel("p")
        plt.yticks(np.arange(self.resolution), (np.round(self.ps, 2)))
        plt.xticks(np.arange(self.resolution), np.round(self.gammas, 2), rotation='vertical')
        plt.title(f"Posterior over $p$ and $\gamma$\nProb. after {episode} round(s).")


        #Show true values.
        if param_values:

            index_p_true = (np. abs(self.ps - param_values.p)). argmin()
            index_gamma_true = (np. abs(self.gammas - param_values.gamma)). argmin()


            #Show probability of true parameters.
            if show_true_prob:
                prob_true = self.prob_true(episode=episode, true_params=param_values)
                plt.plot(index_gamma_true, index_p_true, "D", label = f"True: (p,$\gamma$)={param_values.p,param_values.gamma}" + r", $\mathbb{P}$" + f"=({round(prob_true[0],2), round(prob_true[1],2)})")

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


    def plot_statistics_over_time(self,
                                  episode: int,
                                true_params: ParamTuple):
        
        '''
        Plot the mean and MAP over episodes.
        '''    

        #Statistics to plot.
        episodes = np.arange(episode+1)
        mean_per_episode = [self.mean(episode=i) for i in range(episode+1)]
        MAP_per_episode = [self.MAP(episode=i) for i in range(episode+1)]
        prob_true_per_episode = [self.prob_true(episode=i, true_params=true_params) for i in range(episode+1)]
        prob_true_per_episode[0] = (0,0)

        #Extract values.
        mean_p_per_episode = [mean_per_episode[i].p for i in range(episode+1)]
        mean_gamma_per_episode = [mean_per_episode[i].gamma for i in range(episode+1)]
        MAP_p_per_episode = [MAP_per_episode[i].p for i in range(episode+1)]
        MAP_gamma_per_episode = [MAP_per_episode[i].gamma for i in range(episode+1)]
        MAP_p_per_episode[0] = 0.5 #MAP of the prior distribution, so doesn't mean anything. TODO make this clearer.
        MAP_gamma_per_episode[0] = 0.5
        prob_true_p_per_episode = [prob_true_per_episode[i][0] for i in range(episode+1)]
        prob_true_gamma_per_episode = [prob_true_per_episode[i][1] for i in range(episode+1)]


        #Plot statistis.
        # Create figure and plot the statistics.
        fix, axs = plt.subplots(1, 2, figsize=(14, 3))
        axs[0].hlines(true_params.p, 0, episode, colors="red", linestyle='dashed', label=f"True $p = {round(true_params.p,2)}$")
        axs[0].plot(episodes, mean_p_per_episode, "x-", color="red", label="Mean $p$")
        axs[0].plot(episodes, MAP_p_per_episode, "o-", color="red", label="MAP $p$")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Value")
        axs[0].set_title("Statistics over Time for $p$ (red) and $\gamma$ (blue).")
        axs[0].set_xticks(np.arange(0, episode+1, 1.0))

        axs[0].hlines(true_params.gamma, 0, episode, colors="blue", linestyle='dashed', label=f"True $\gamma = {round(true_params.gamma,2)}$")
        axs[0].plot(episodes, mean_gamma_per_episode, "x-", color="blue", label="Mean $\gamma$")
        axs[0].plot(episodes, MAP_gamma_per_episode, "o-", color="blue", label="MAP $\gamma$")
        axs[0].legend(loc="lower right", ncol=2)


        axs[1].plot(episodes, prob_true_p_per_episode, "x-", color = "red", label="Prob. $p$ true")
        axs[1].plot(episodes, prob_true_gamma_per_episode, "o-", color = "blue", label="Prob. $\gamma$ true")
        axs[1].legend(loc="lower right")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Probability")
        axs[1].set_title("Probability of True Values $p$ (red) and $\gamma$ (blue) over Time.")
        axs[1].set_xticks(np.arange(0, episode+1, 1.0))

        plt.tight_layout()
        plt.show()


    def mean(self,
             episode:int = None,
             posterior_dist: np.array = None):

        assert (episode is not None) or (posterior_dist is not None), f"Supply either episode to calculate mean for or supply posterior distribution." 

        if episode is not None:
            self._validate_episode(episode=episode)

            #Arrays to take mean over.
            posterior_probabilities = np.exp(self.posterior_distribution[f"episode={episode}"])

        elif posterior_dist is not None:

            posterior_probabilities = np.exp(posterior_dist)

        total_probability = np.sum(posterior_probabilities, axis=tuple(range(posterior_probabilities.ndim)))
        
        #Calculate mean.
        _means = []
        for i in range(posterior_probabilities.ndim):
            _indexes_to_sum = [j for j in range(posterior_probabilities.ndim) if j != i]
            _probs = np.sum(posterior_probabilities, axis=tuple(_indexes_to_sum))
            _means.append(np.sum(self.parameter_ranges[i]*_probs)/total_probability)

        return _means
    

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

        total_probability = np.sum(posterior_distribution, axis=(0,1))

        #Uniform prior.
        if total_probability == 0:
            return (0,0)
        else:
            prob_p_true = np.sum(posterior_distribution[index_p_true,:])/total_probability
            prob_gamma_true = np.sum(posterior_distribution[:,index_gamma_true])/total_probability
            
            return (prob_p_true, prob_gamma_true)
        
    
    def calculate_region_of_interest(self,
                           log_likelihood, 
                           confidence_interval: float = 0.8
                           ):

        assert (confidence_interval >= 0) and (confidence_interval <= 1), f"Confidence interval must be in [0,1], you gave value {confidence_interval}."

        #Convert array from log-likelihood to likelihood and normalize to 1.
        flat_log_likelihood = log_likelihood.flatten()
        flat_likelihood = np.exp(flat_log_likelihood)
        flat_likelihood = flat_likelihood/np.sum(flat_likelihood)

        n = 0
        current_mass = 0


        sorted_indexes = np.argsort(flat_likelihood)[::-1]

        for i in range(len(sorted_indexes)):
            current_mass += flat_likelihood[sorted_indexes[i]]
            if current_mass > confidence_interval:
                n = i + 1
                break

        #Change to Numpy Array for easier indexing
        region_of_interest = sorted_indexes[:n]

        return region_of_interest