from typing import List

from tqdm import tqdm
import numpy as np

from .make_environment import transition_matrix, insert_walls_into_T
from .optimization import soft_q_iteration, grad_policy_maximization, value_iteration_with_policy
from .inference.rollouts import generate_n_trajectories
from .inference.likelihood import compute_log_likelihood
from .make_environment import Environment
from .constants import ParamTuple
from .constants import beta_agent


class EnvironmentDesign():

    '''
    
    '''

    def __init__(self,
                 base_environment: Environment,
                 user_params: ParamTuple):
        
        self.base_environment = base_environment
        self.user_params = user_params

    
    def run_n_episodes(self,
                       n_episodes: int):
        
        '''
        Run Environment Design for n_episodes episodes.
        '''
        
        self.episodes = n_episodes


    def generate_candidate_environments(self,
                                        num_candidate_environments: int,
                                        generate_how: str,
                                        random_envs_specs: dict):
        
        '''
        Generate candidate environments for the Bayesian Regret calculation.
        '''
        
        self.num_candidate_environments = num_candidate_environments
        generate_how = generate_how


        if generate_how == "random_walls":


            #Number of walls to insert.
            n_walls = random_envs_specs["n_walls"]


            #Generate copies of base enviroment.
            candidate_envs = [
                Environment(
                    N=self.base_environment.N,
                    M=self.base_environment.M,
                    T_true=self.base_environment.T_true,
                    goal_states=self.base_environment.goal_states,
                    wall_states=self.base_environment.wall_states,
                    n_walls=self.base_environment.n_walls,
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

            self.candidate_environments = candidate_envs


    def observe_human(self,
                      environment: Environment,
                      n_trajectories: int=2):
        
        '''
        Observe human in an environment n_trajectories times.

        Args:
        - environment: environment in which we observe the human.
        - n_trajectories: number of times we observe the human.

        Returns:
        - list of [Environment, trajectories]
        '''
        

        #Calculate policy of agent in environment.
        T_agent = transition_matrix(environment.N, environment.M, p=self.user_params.p, absorbing_states=environment.goal_states)
        T_agent = insert_walls_into_T(T=T_agent, wall_indices=environment.wall_states)
        agent_policy = soft_q_iteration(self.user_params.R, T_agent, gamma=self.user_params.gamma, beta=1000)

        # Generate trajectories
        trajectories = generate_n_trajectories(
            environment.T_true,
            agent_policy,
            environment.goal_states,
            n_trajectories=n_trajectories,
        )

        return [(environment, trajectories)]





        

        







def environment_search(
    N,
    M,
    goal_states,
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
    N, M: width and length of environment
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
    n_states = N*M

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
                T_agent = transition_matrix(N, M, p=p, absorbing_states=goal_states)
                T_agent = insert_walls_into_T(T_agent, wall_indices=candidate_env.wall_states)
                policy = soft_q_iteration(R, T_agent, gamma=gamma, beta=1000)
                policies.append(policy)

                # 4.1.2 Generate $m$ trajectories from this policy

                policy_traj = generate_n_trajectories(
                    candidate_env.T_true,
                    policy,
                    goal_states,
                    start_state=candidate_env.start_state,
                    n_trajectories=n_traj_per_sample,
                    # Walking from the top-left to the bottom-right corner takes at most N + M - 2 steps
                    # so we allow twice this at most
                    max_steps=(N + M - 2) * 2,
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
            # raise Exception("STOP")

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
            R_sample_mean = np.mean([sample[2] for sample in posterior_samples], axis=0)
            candidate_env.R_sample_mean = R_sample_mean
            del R_sample_mean

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
                    p = agent_p
                if agent_gamma is not None:
                    gamma = agent_gamma
                if agent_R is not None:
                    R = agent_R

                #agents transition function according to p_sample
                T_agent = transition_matrix(N, M, p=p_sample, absorbing_states=goal_states)
                T_agent = insert_walls_into_T(T_agent, wall_indices=candidate_env.wall_states)

                V, _ = value_iteration_with_policy(R_sample, T_agent, gamma_sample)
                regret += V[0] / len(posterior_samples)
                # print("regret: ", regret)

            # calculate regret for one policy across all samples
            R_sample_mean = np.mean([sample[2] for sample in posterior_samples], axis=0)
            p_sample_mean = np.mean([sample[1] for sample in posterior_samples], axis = 0)
            gamma_sample_mean = np.mean(
                [sample[0] for sample in posterior_samples], axis=0
            )

            T_agent_mean = transition_matrix(N, M, p=p_sample_mean, absorbing_states=goal_states)
            T_agent_mean = insert_walls_into_T(T_agent_mean, wall_indices=candidate_env.wall_states)
            V_mean, _ = value_iteration_with_policy(
                R_sample_mean, T_agent_mean, gamma_sample_mean
            )

            regret -= V_mean[0]
            candidate_env.regret = regret

            # 4.4 If the regret is higher than the highest regret so far, store the env and policy
            if candidate_env.regret > highest_regret:
                highest_regret = candidate_env.regret
                pbar.set_postfix({"highest_regret": highest_regret})

            candidate_env.id = candidate_env_id
            candidate_env_id += 1
            candidate_env.R_sample_mean = R_sample_mean

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