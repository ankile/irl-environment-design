from typing import List

import os
from tqdm import tqdm
import numpy as np
import pickle
import datetime
from copy import deepcopy

from .make_environment import transition_matrix, insert_walls_into_T
from .optimization import soft_q_iteration, grad_policy_maximization, value_iteration_with_policy
from .inference.rollouts import generate_n_trajectories
from .inference.likelihood import compute_log_likelihood
from src.utils.inference.sampling import bayesian_parameter_learning
from .make_environment import Environment
from .constants import ParamTuple
from .inference.posterior import PosteriorInference
from src.utils.make_candidate_environments import EntropyBM
from src.worlds.mdp2d import Experiment_2D


class EnvironmentDesign():

    '''
    TODO
    '''

    def __init__(self,
                 base_environment: Environment,
                 user_params: ParamTuple,
                 learn_what: list):
        
        '''
        
        Args:
        - learn_what: list: which parameters we learn, e.g. learn_what = ['R', 'gamma'] means we learn R and gamma while the transition function is assumed to be known.
        '''
        
        self.base_environment = base_environment
        self.user_params = user_params
        self.all_observations = []
        self.learn_what = learn_what

        self.candidate_env_generation_methods = ["random_walls", "hard_coded_envs"]

    
    def run_n_episodes(self,
                       n_episodes: int,
                       candidate_environments_args: dict,
                       bayesian_regret_how = None,):
        
        '''
        Run Environment Design for n_episodes episodes.
        An episode is defined as observing the agent in an environment. We always observe the agent in the base environment first.
        Then, we perform Environment Design. Thereby, if n_episodes = n, we perform Environment Design n-1 times. 

        Args:
        - n_episodes: number of episodes to run environment design for.
        - bayesian_regret_how: how to evaluate the Bayesian Regret. Supported methods: ['value', 'likelihood'].
        - candidate_environments_args: dict for the respective candidate generation method.
        '''
        
        self.episodes = n_episodes
        self.candidate_environments_args = candidate_environments_args

        #Observe human in base environment. Append observation to all observations.
        print("Started episode 0.")
        observation = self._observe_human(environment=self.base_environment, n_trajectories=2)
        self.all_observations.append(observation)
        print("Finished episode 0.")
        region_of_interest = None

        for episode in range(1,self.episodes):
        
            print(f"Started episode {episode}.")

            if candidate_environments_args["generate_how"] == "entropy_BM":


                #TODO min/ max values need to be inferred from ROI. Are inferred but make this cleaner, e.g. "zoom in" on BM.
                min_gamma = 0.7
                max_gamma = 0.99
                min_p = 0.7
                max_p = 0.99
                pos_inference = PosteriorInference(self.all_observations,
                                                   resolution=10,
                                                   min_gamma = min_gamma,
                                                   max_gamma = max_gamma,
                                                   min_p = min_p,
                                                   max_p = max_p,
                                                   region_of_interest=region_of_interest)
                
                print("Started computing Posterior.")
                current_belief = pos_inference.calculate_posterior(episode=episode)
                print("current_belief:", current_belief)
                print("Finished computing Posterior.")
                map_params = pos_inference.mean(posterior_dist = current_belief) #TODO change this to MAP.
                region_of_interest = pos_inference.calculate_region_of_interest(log_likelihood = current_belief, confidence_interval=0.8)
                print("Region of Interest:", region_of_interest)

                print(f"Computed Region of Interest. Size = {round(region_of_interest.size/current_belief.size, 2)}")


                #TODO here we need to have a cleaner way to convert the parametrization into the actual function.
                if "R" in self.learn_what:
                    R_estimate = map_params.R
                else:
                    R_estimate = self.user_params.R


                if "gamma" in self.learn_what:
                    gamma_estimate = map_params.gamma
                else:
                    gamma_estimate = self.user_params.gamma


                if "T" in self.learn_what:
                    T_estimate = transition_matrix(self.base_environment.N, self.base_environment.M, p=map_params.p, absorbing_states=self.base_environment.goal_states)
                    T_estimate = insert_walls_into_T(T=T_estimate, wall_indices=self.base_environment.wall_states)
                else:
                    T_estimate = self.base_environment.T_true


                param_estimates = ParamTuple(p = T_estimate, gamma=gamma_estimate, R = R_estimate)


                #Initialize EntropyBM object.
                entropy_bm = EntropyBM(parameter_estimates=param_estimates,
                                       gammas = np.linspace(min_gamma, max_gamma, num=15),
                                       probs= np.linspace(min_p, max_p, num=15),
                                       region_of_interest=region_of_interest,
                                       )
                
                #World to compute Behavior Map. TODO: this should take arbitrary arguments and not only gamma/p.
                _world = Experiment_2D(self.base_environment.N,
                                       self.base_environment.M,
                                       rewards=R_estimate,
                                       absorbing_states=self.base_environment.goal_states,
                                       wall_states=self.base_environment.wall_states)

                #Find a reward function that maximizes the entropy of the Behavior Map. TODO: also use transition function. Currently only do gradient updates on R.
                updated_reward = entropy_bm.BM_search(world = _world,
                                                      n_compute_BM = 5,
                                                      n_iterations_gradient=20,
                                                      stepsize_gradient=0.001)
                
                print("Learned Reward Function that maximizes Entropy. Reward function: ", updated_reward)
                
                #Generate an environment in which we observe the human with maximal information gain.
                optimal_environment = deepcopy(self.base_environment)
                optimal_environment.R_true = updated_reward
                
                




            elif candidate_environments_args["generate_how"] in ["random_walls", "hard_coded_envs"]:

                #Generate Candidate Environments.
                candidate_environments = self._generate_candidate_environments(num_candidate_environments=candidate_environments_args["n_environments"],
                                                    generate_how=candidate_environments_args["generate_how"],
                                                    candidate_env_specs=candidate_environments_args)
                
                #Generate Samples from current belief.
                samples = self._sample_posterior(observations=self.all_observations,
                                                sample_size=250,
                                                burnin=150)

                #Find maximum Bayesian Regret environment.
                candidate_environments_sorted = self._environment_search(base_environment=self.base_environment,
                                        posterior_samples=samples,
                                        n_traj_per_sample=1,
                                        candidate_envs=candidate_environments,
                                        how=bayesian_regret_how
                                        )
                
                del samples
                del candidate_environments
                
                #Maximum Regret environment
                optimal_environment = candidate_environments_sorted[0]

                del candidate_environments_sorted
            
            #Observe human in environment. Append observation to all observations.
            observation = self._observe_human(environment=optimal_environment,n_trajectories=1)
            self.all_observations.append(observation)

            del observation
            print(f"Finished episode {episode}.")



    def save(self, experiment_name: str):
        '''
        Save all relevant information of the EnvironmentDesign object to a file.

        Args:
        - experiment_name: name of the file to save the information to.
        '''
        todays_date = datetime.date.today().strftime('%d.%m.%Y')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        data = {
            'base_environment': self.base_environment,
            'user_params': self.user_params,
            'all_observations': self.all_observations,
            'episodes': self.episodes,
            'candidate_environment_args': self.candidate_environments_args
        }

        filepath = os.path.join(os.getcwd(), "checkpoints", experiment_name, todays_date)
        filename = current_time
        os.makedirs(filepath, exist_ok=True)

        with open(os.path.join(filepath, filename), 'wb+') as file:
            pickle.dump(data, file)

        del data, filepath, todays_date, current_time


    #TODO, this should be in make_environment.py, not here.
    def _generate_candidate_environments(self,
                                        num_candidate_environments: int,
                                        generate_how: str,
                                        candidate_env_specs: dict):
        
        '''
        Generate candidate environments for the Bayesian Regret calculation.
        '''
        
        self.num_candidate_environments = num_candidate_environments
        generate_how = generate_how


        if generate_how == "random_walls":


            #Number of walls to insert.
            n_walls = candidate_env_specs["n_walls"]


            #Generate copies of base enviroment.
            candidate_envs = [
                Environment(
                    N=self.base_environment.N,
                    M=self.base_environment.M,
                    T_true=self.base_environment.T_true,
                    goal_states=self.base_environment.goal_states,
                    wall_states=self.base_environment.wall_states,
                    n_walls=self.base_environment.n_walls,
                    R_true=self.base_environment.R_true
                )
                for _ in range(self.num_candidate_environments)
            ]

            #Insert random walls into candidate environments.
            for candidate_env in candidate_envs:

                # Generate walls at random locations.
                wall_incides = np.random.randint(1, self.base_environment.N*self.base_environment.M-1, size=n_walls)
                #Add existing walls
                wall_incides = np.append(wall_incides, candidate_env.wall_states)

                #Remove potential wall in start state.
                wall_incides = np.setdiff1d(wall_incides, self.base_environment.start_state)

                #Update transition matrix.
                #TODO we add wall states of initial environment twice. Check whether this is a problem.
                candidate_env.T_true = insert_walls_into_T(
                    candidate_env.T_true, wall_indices=wall_incides
                )

                #Append wall to list of walls of candidate environment.
                candidate_env.wall_states = wall_incides

            return candidate_envs
        
        elif generate_how == "hard_coded_envs":

            return candidate_env_specs["candidate_envs"]
        
        elif generate_how == "Naive_BM":
            
        

        else:
        
            raise NotImplementedError(f"Candidate Environment generation method not implemented. Supported methods are: {self.candidate_env_generation_methods}")
    


    def _observe_human(self,
                      environment: Environment,
                      n_trajectories: int=2):
        
        '''
        Observe human in an environment n_trajectories times.
        Args:
        - environment: environment in which we observe the human.
        - n_trajectories: number of times we observe the human.

        Returns:
        - tuple of (Environment, trajectories)
        '''
        

        #Calculate policy of agent in environment.
        T_agent = transition_matrix(environment.N, environment.M, p=self.user_params.p, absorbing_states=environment.goal_states)
        T_agent = insert_walls_into_T(T=T_agent, wall_indices=environment.wall_states)
        agent_policy = soft_q_iteration(self.user_params.R, T_agent, gamma=self.user_params.gamma, beta=1000)

        # Generate trajectories.
        trajectories = generate_n_trajectories(
            environment.T_true,
            agent_policy,
            environment.goal_states,
            n_trajectories=n_trajectories,
        )

        del agent_policy
        del T_agent

        return (environment, trajectories)


    def _sample_posterior(self,
                          observations,
                          sample_size: int = 500,
                          burnin: int = 250
                          ):
        
        assert sample_size > burnin, f"Burnin can't be larger than sample size. You set burnin={burnin}, sample_size={sample_size}"
        
        #Generate samples from posterior via Metropolis Hastings.
        _samples = bayesian_parameter_learning(
            expert_trajectories=observations,
            sample_size=sample_size
        )

        #Take burnin away.
        _n_samples_to_keep = sample_size - burnin
        return _samples[-_n_samples_to_keep:]






    def _environment_search(
        self,
        base_environment: Environment,
        posterior_samples,
        n_traj_per_sample,
        candidate_envs,
        n_actions = 4,
        how = "likelihood",
        return_sorted = True,
        agent_p = None,
        agent_gamma = None,
        agent_R = None
    ):
        """
        how: use likelihood or value function to measure regret, in ["likelihood", "value"]
        goal_states: numpy array of (absorbing) goal states
        n_env_samples: how many candidate environments to generate
        posterior_samples: samples from posterior
        n_traj_per_sample: number of trajectories to generate for each sample, only relevant if how == "likelihood"
        candidate_envs: list of environments for which we calculate the regret
        agent_p: agent perceived transition rate, if given is used instead of the samples
        agent_gamma: agent gamma, if given is used instead of the samples
        agent_R: agent R, if given is used instead of the samples
        """
        n_states = base_environment.N*base_environment.M

        # 1. Initialize storage
        highest_regret = -np.inf

        pbar = tqdm(
            candidate_envs,
            desc=f"Evaluating candidate environments using {how}",
            postfix={"highest_regret": highest_regret},
        )

        if how == "likelihood":
            '''
            Use the log-likelihood for the Bayesian Regret calculation.
            '''
            candidate_env_id = 0

            for candidate_env in pbar:
                policies = []
                trajectories = []
                likelihoods = []

                for p, gamma, R in posterior_samples:

                    #if we dont want to learn some parameter, we overwrite the sample with the true value
                    if agent_p is not None:
                        p = agent_p
                    if agent_gamma is not None:
                        gamma = agent_gamma
                    if agent_R is not None:
                        R = agent_R

                    # 4.1.1 Find the optimal policy for this env and posterior sample
                    T_agent = transition_matrix(candidate_env.N, candidate_env.M, p=p, absorbing_states=candidate_env.goal_states)
                    T_agent = insert_walls_into_T(T_agent, wall_indices=candidate_env.wall_states)
                    policy = soft_q_iteration(R, T_agent, gamma=gamma, beta=1000)
                    policies.append(policy)

                    # 4.1.2 Generate $m$ trajectories from this policy

                    policy_traj = generate_n_trajectories(
                        candidate_env.T_true,
                        policy,
                        candidate_env.goal_states,
                        start_state=candidate_env.start_state,
                        n_trajectories=n_traj_per_sample,
                        # Walking from the top-left to the bottom-right corner takes at most N + M - 2 steps
                        # so we allow twice this at most
                        max_steps=(candidate_env.N + candidate_env.M - 2) * 2,
                    )


                    # 4.1.3 Calculate the likelihood of the trajectories
                    policy_likelihoods = [
                        compute_log_likelihood(candidate_env.T_true, policy, traj)
                        for traj in policy_traj
                    ]

                    # 4.1.4 Store the trajectories and likelihoods
                    trajectories += policy_traj
                    likelihoods += policy_likelihoods

                # 4.2 Find the policy with the highest likelihood
                most_likely_policy = grad_policy_maximization(
                    n_states=n_states,
                    n_actions=n_actions,
                    trajectories=trajectories,
                    T_true=candidate_env.T_true,
                    n_iter=100,
                )
                candidate_env.max_likelihood_policy = most_likely_policy

                # 4.3 Calculate the regret of the most likely policy
                most_likely_likelihoods = [
                    compute_log_likelihood(candidate_env.T_true, most_likely_policy, traj)
                    for traj in trajectories
                ]

                all_likelihoods = np.array([likelihoods, most_likely_likelihoods]).T
                candidate_env.log_likelihoods = all_likelihoods.mean(axis=0)
                candidate_env.log_regret = -np.diff(candidate_env.log_likelihoods).item()

                # all_likelihoods = np.exp(all_likelihoods)
                # candidate_env.likelihoods = all_likelihoods.mean(axis=0)
                # candidate_env.regret = -np.diff(candidate_env.likelihoods).item() #there was a "-" in front of np.diff here. Do you know why?

                candidate_env.trajectories = trajectories

                # 4.4 If the regret is higher than the highest regret so far, store the env and policy
                if candidate_env.log_regret > highest_regret:
                    highest_regret = candidate_env.log_regret
                    pbar.set_postfix({"highest_log_regret": highest_regret, "wall_states": candidate_env.wall_states})
                candidate_env.id = candidate_env_id
                candidate_env_id += 1

                # add reward sample mean to environment for visualization
                # R_sample_mean = np.mean([sample[2] for sample in posterior_samples], axis=0)
                # candidate_env.R_sample_mean = R_sample_mean
                # del R_sample_mean

            # 5. Return the environments (ordered by regret, with higest regret first)
            if return_sorted:
                return sorted(candidate_envs, key=lambda env: env.log_regret, reverse=True)
            else:
                return candidate_envs

        elif how == "value":
            """
            Environment Design using the Value Function to calculate the Bayesian Regret as in the original Environment Design Paper
            """

            candidate_env_id = 0

            for candidate_env in pbar:
                regret = 0


                # calculate regret for one policy for each sample
                for p_sample, gamma_sample, R_sample in posterior_samples:

                    #if we dont want to learn some parameter, we overwrite the sample with the true value
                    if agent_p is not None:
                        p_sample = agent_p
                    if agent_gamma is not None:
                        gamma_sample = agent_gamma
                    if agent_R is not None:
                        R_sample = agent_R

                    #agents transition function according to p_sample
                    T_agent = transition_matrix(candidate_env.N, candidate_env.M, p=p_sample, absorbing_states=candidate_env.goal_states)
                    T_agent = insert_walls_into_T(T_agent, wall_indices=candidate_env.wall_states)
                    V, _ = value_iteration_with_policy(candidate_env.R_true, T_agent, gamma_sample)
                    regret += V[0] / len(posterior_samples)
                    # print("regret: ", regret)

                # calculate regret for one policy across all samples
                # R_sample_mean = np.mean([sample[2] for sample in posterior_samples], axis=0)
                p_sample_mean = np.mean([sample[1] for sample in posterior_samples], axis = 0)
                gamma_sample_mean = np.mean(
                    [sample[0] for sample in posterior_samples], axis=0
                )

                T_agent_mean = transition_matrix(candidate_env.N, candidate_env.M, p=p_sample_mean, absorbing_states=candidate_env.goal_states)
                T_agent_mean = insert_walls_into_T(T_agent_mean, wall_indices=candidate_env.wall_states)
                V_mean, _ = value_iteration_with_policy(
                    candidate_env.R_true, T_agent_mean, gamma_sample_mean
                )

                regret -= V_mean[0]
                candidate_env.regret = regret

                # 4.4 If the regret is higher than the highest regret so far, store the env and policy
                if candidate_env.regret > highest_regret:
                    highest_regret = candidate_env.regret
                    pbar.set_postfix({"highest_regret": highest_regret})

                candidate_env.id = candidate_env_id
                candidate_env_id += 1
                # candidate_env.R_sample_mean = R_sample_mean

            # 5. Return the environments (ordered by regret, with higest regret first)
            if return_sorted:
                return sorted(candidate_envs, key=lambda env: env.regret, reverse=True)
            else:
                return candidate_envs
            

        # Non-implemented Bayesian Regret calculation method.
        else:
            raise ValueError(
                f"'how' should be in ['likelihood', 'value'] while you set how = {how}."
            )