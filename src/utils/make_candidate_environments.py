from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import src.utils.behavior_map as bm
from src.worlds.mdp2d import Experiment_2D
from src.utils.make_environment import insert_walls_into_T
from src.utils.optimization import differentiate_V



'''
Functions to generate candidate environments.
'''



def make_world(
    height: int,
    width: int,
    rewards: np.array,
    absorbing_states: list,
    wall_states: list
) -> Experiment_2D:
    
    '''
    Convert a 2D gridworld into an Experiment_2D object.
    '''

    experiment = Experiment_2D(
        height,
        width,
        rewards = rewards,
        absorbing_states = absorbing_states,
        wall_states=wall_states,
    )

    return experiment


class EntropyBM():

    '''
    Maximize the Entropy of the Behavior Map via Implicit Differentiation.

    Args:
    - parameter_estimates (namedtuple): The parameter estimates of the MDP. Contains estimates for R, \gamma, T.
    - gammas (list): The parameter bounds for the discount factor.
    - probs (list): The parameter bounds for the probability of the agent choosing the optimal action.
    - region_of_interest (list): The region of interest in the Behavior Map.
    '''

    def __init__(self, 
                 estimate_R,
                 estimate_gamma,
                 estimate_T, 
                 named_parameter_mesh,
                 shaped_parameter_mesh,
                #  gammas, 
                #  probs, 
                 region_of_interest,
                 verbose) -> None:

        self.estimate_R = estimate_R
        self.estimate_gamma = estimate_gamma
        self.estimate_T = estimate_T

        self.named_parameter_mesh = named_parameter_mesh
        self.shaped_parameter_mesh = shaped_parameter_mesh
        self.region_of_interest = region_of_interest
        self.verbose = verbose


    def compute_covers(self, behavior_map):

        '''
        Computes the entropy of the Behavior Map and for each behavior in the behavior map the proportion of the BM that is covered by the respective Behavior.

        Args:
        - bm_out (dict): The output of the behavior map function.

        Returns:
        - covers (dict): A dictionary containing the proportion of the BM that is covered by the respective Behavior.
        - max_ent_cover (float): The proportion of the BM that would be covered by each behavior in the case of maximum entropy.
        '''

        #Compute Behavior Map restricted to Region of Interest.
        # behavior_ROI = self.compute_bm_ROI(behavior_map)

        behavior_ROI = behavior_map.data.flatten()

        _behaviors = np.unique(behavior_ROI)
        n_behavior_samples = len(behavior_ROI)
        covers = {}

        for b in _behaviors:
            covers[b] = np.sum(behavior_ROI == b) / n_behavior_samples

        max_ent_cover = 1/len(_behaviors)

        return covers, max_ent_cover
    

    def gradient_updates_R(self, R_init, bm_out, stepsize: float = 0.001, n_iterations: int = 20):

        '''
        Perform gradient updates on the reward function R to maximize the entropy of the behavior map.

        Args:
        - world (Experiment_2D): The world in which the agent is acting.
        - bm_out (dict): The output of the behavior map function.
        - stepsize (float): The stepsize of the gradient updates.
        - n_iterations (int): The number of iterations to run.

        Returns:
        - R (torch.tensor): The updated reward function R.
        '''

        covers, max_ent_cover = self.compute_covers(bm_out)

        # Initialize learning parameters
        R = torch.tensor(R_init, dtype=torch.float32)
        gamma = torch.tensor(self.estimate_gamma, dtype=torch.float32)
        T = torch.tensor(self.estimate_T, dtype=torch.float32)
        V_star = torch.zeros_like(R)

        for _ in range(n_iterations):


            # Compute the gradient of the value function with respect to the reward function and the transition matrix.
            V_star, R_grad_out, _ = differentiate_V(R = R, gamma = gamma, T = T, V = V_star)

            # Update the reward function
            for behavior_idx in covers:

                #Get states that are visited by the behavior.
                cover = covers[behavior_idx]
                _visited_states = bm_out.pidx2states[behavior_idx]

                #TODO: think about whether we want the next line or not.
                # _visited_states = _visited_states[:-1]
                
                #Get gradient along those visited states.s
                _masked_gradient_R = torch.zeros_like(R_grad_out)
                _masked_gradient_R[_visited_states] = R_grad_out[_visited_states]

                # print("behavior_idx: ", behavior_idx)
                # print("Cover: ", cover)
                # print("Visited States: ", _visited_states)
                # print("Gradient: ", _masked_gradient_R)

                #Inhibit behavior that covers more than maximum entropy share.
                if (cover > max_ent_cover) or (cover == 1):
                    R = R - stepsize * _masked_gradient_R

                #Excite behavior that covers less than maximum entropy share.
                else:
                    R = R + stepsize * _masked_gradient_R

        return R
    
    
    def BM_search(self, base_environment, named_parameter_mesh, shaped_parameter_mesh, n_compute_BM: int, n_iterations_gradient: int = 50, stepsize_gradient: float = 0.025):

        '''
        Find a reward function that maximizes the entropy of the Behavior Map.

        Args:
        TODO    
        '''

        environment = deepcopy(base_environment)
        R = self.estimate_R
        _max_ent = -np.inf
        _max_ent_cover = None
        max_ent_R = self.estimate_R
        R_entropy_update = np.zeros_like(R)
        region_of_interest = self.region_of_interest

        entropy_maximized: bool = False
        n_iterations = 0

        # for _ in range(n_compute_BM):
        while not entropy_maximized:


            # Compute Behavior Map
            bm_out = bm.calculate_behavior_map(environment=environment,
                                               reward_update=R_entropy_update,
                                               parameter_mesh=named_parameter_mesh,
                                               shaped_parameter_mesh=shaped_parameter_mesh,
                                               region_of_interest = region_of_interest)
            
            # print("Behavior Map:", bm_out)

            #Compute entropy of BM
            cover, max_ent_prob = self.compute_covers(bm_out)
            entropy_BM = stats.entropy(list(cover.values()))

            # print("Cover: ", cover)

            #Check if the current Behavior Map has higher entropy.
            if entropy_BM > _max_ent:
                max_ent_possible = stats.entropy(np.repeat(max_ent_prob, repeats=int(1/max_ent_prob)))
                _max_ent = entropy_BM
                _max_ent_cover = cover
                max_ent_R = R
                _max_ent_BM = bm_out

            # Perform Gradient Updates on Reward Function to maximize entropy of BM.
            # print("\nUpdating Reward Function\n")
            R_entropy_update = self.gradient_updates_R(R_init = R, bm_out=bm_out, stepsize=stepsize_gradient, n_iterations=n_iterations_gradient)
            R_entropy_update = R_entropy_update.detach().numpy()

            #Update Reward Function
            R = R_entropy_update

            #TODO: should we save the previous maximum entropy reward function for the next iteration to warm start?
            #Save Reward Function
            # environment.reward_function = R_update.detach().numpy()

            #Check if the entropy of the Behavior Map has been maximized.
            if (np.isclose(_max_ent, max_ent_possible, atol = 0.01)) and max_ent_possible != 0:
                entropy_maximized = True

            n_iterations += 1

            if n_iterations > n_compute_BM:
                print("\n\nReached Maximum Number of BM Computations. Terminating BM Search.\n\n")
                break

        if self.verbose:
            print(f"Finished BM Search. Entropy: {_max_ent}. Max Ent possible: {max_ent_possible}. Cover: {_max_ent_cover}. Behaviors: {bm_out.pidx2states}")
            print("Behavior map: ", _max_ent_BM)
            print("Reward Function: ", max_ent_R)

        return max_ent_R
        


class AgnosticsBM():

    '''
    Calculate the Behavior Map of an environment and perturb the environment such that the dominant policy becomes less attractive 
    and the subdominant policies become more attractive.
    '''

    def __init__(self, 
                 environment,
                 behavior_map_environment,
                 region_of_interest=None):
        
        self.environment = environment
        self.perturbed_environment = deepcopy(self.environment)
        self.region_of_interest  = region_of_interest
        self.behavior_map_base_environment = behavior_map_environment
        self.behavior_map_perturbed_environment = None
        self.prop_dominant_policy = []
        self.prop_subdominant_policy = []
        self.prop_unreasonable_policy = []
        self.perturbed_environments = {}
        self.perturbed_behavior_maps = {}
        self.n_accepted = 0

    def calculate_behavior_map_stats(self, behavior_map):

        '''
        Determine reasonable, unreasonable and dominant policies in the behavior map.
        '''

        #Reasonable policies are all policies that don't stay in the start state. Unreasonable policies stay in the start state.
        reasonable_policies_idx = [p for p, states in behavior_map.pidx2states.items() if states[-1] in self.environment.goal_states]
        unreasonable_policies_idx = [p for p, states in behavior_map.pidx2states.items() if states[-1] not in self.environment.goal_states]

        #Number of different policies.
        behaviors_flattened = behavior_map.data.flatten()
        #Only get behaviors in Region of interest.
        if self.region_of_interest is not None:
            
            behaviors_flattened = behavior_map.data[tuple(self.region_of_interest.T)]
        n_behaviors = len(behaviors_flattened)

        if self.region_of_interest is not None:
            assert n_behaviors == self.region_of_interest.shape[0], "The number of behaviors in the region of interest does not match the number of behaviors in the behavior map."

        #Number of reasonable policies.
        n_reasonable_behaviors = np.sum(np.isin(behaviors_flattened, reasonable_policies_idx))
        n_unreasonable_behaviors = np.sum(np.isin(behaviors_flattened, unreasonable_policies_idx))

        #Count the number of times each policy is chosen.
        behavior, counts = np.unique(behaviors_flattened, return_counts=True)
        behavior_counts = dict(zip(behavior, counts))

        #Dominant policy is the policy with the most counts, e.g. that covers the largest proportion of the Behavior Map.
        # Remove all unreasonable policies from the counts.
        for unreasonable_policy in unreasonable_policies_idx:
            behavior_counts.pop(unreasonable_policy, None)

        #Get the dominant policy. If there are only unreasonable policies (e.g. behavior counts ie empty), then the dominant policy is None.
        if behavior_counts == {}:
            dominant_policy = None
            return 0, 0, 1, None
        else:
            dominant_policy = max(behavior_counts, key=behavior_counts.get)

        #Subdominant policies are all policies that are a) not dominant and b) not unreasonable.
        n_dominant_behaviors = behavior_counts[dominant_policy]
        n_subdominant_behaviors = n_reasonable_behaviors - n_dominant_behaviors

        #Determine proportion of BM that each policy type covers.
        prop_dominant_policy = n_dominant_behaviors / n_behaviors
        prop_subdominant_policy = n_subdominant_behaviors / n_behaviors
        prop_unreasonable_policy = n_unreasonable_behaviors / n_behaviors

        #Get a rollout of the dominant policy.
        dominant_states = behavior_map.pidx2states[dominant_policy]

        return prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states

    def perturb_transition_dynamics(self, states):

        '''
        We want to change the transition dynamics, such that prop_dominant_policy decreases and prop_subdominant_policy increases while prop
        _unreasonable_policy remains small.
        We can do this by changing the reward function and transition dynamics such that the dominant policy becomes less attractive and the subdominant
        poicies become more attractive. To this end, we insert walls/ death states along the rollouts of the dominant policy and remove walls/death states
        from the rollouts of the subdominant policies. This will make the dominant policy less attractive and the subdominant policies more attractive.
        '''
        #Get a random state from the dominant policy to insert a wall into. Remove goal states and start state.
        random_state_from_dominant = np.random.choice(list(states), size = 1)
        random_state_from_dominant = np.setdiff1d(random_state_from_dominant, self.environment.goal_states)
        #TODO: start state should be flexible, not 0.
        random_state_from_dominant = np.setdiff1d(random_state_from_dominant, [0])

        #Insert walls into the transition matrix along the rollouts of the dominant policy and update transition function.
        T_new = insert_walls_into_T(T=self.perturbed_environment.T_true, wall_indices=random_state_from_dominant)
        self.perturbed_environment.wall_states = np.append(self.perturbed_environment.wall_states, random_state_from_dominant)
        self.perturbed_environment.T_true = T_new
        print(f"Perturbed transition dynamics. Inserted a wall into state {random_state_from_dominant}.")

    
    def perturb_reward_function(self, states):

        '''
        Change reward function along dominant rollout of dominant.
        '''
        #Get a random state from the dominant policy to insert a wall into. Remove goal states and start state.
        random_state_from_dominant = np.random.choice(list(states), size = 1)
        random_state_from_dominant = np.setdiff1d(random_state_from_dominant, self.environment.goal_states)
        #TODO: start state should be flexible, not 0.
        random_state_from_dominant = np.setdiff1d(random_state_from_dominant, [0])
        #Insert negative reward into the transition matrix along the rollouts of the dominant policy.
        R_new = self.perturbed_environment.R_true.copy()
        R_new[random_state_from_dominant] += -0.1

        self.perturbed_environment.R_true = R_new
        print(f"Perturbed reward function. Inserted a negative reward into state {random_state_from_dominant}.")

    
    def perturb_environment(self, n_iterations: int,
                            plot_bmap: bool = False,):

        '''
        Perturb the environment such that the dominant policy becomes less attractive and the subdominant policies become more attractive.
        '''

        #Statistics of Base Environment.
        prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states = self.calculate_behavior_map_stats(behavior_map=self.behavior_map_base_environment)
        self.prop_dominant_policy.append(prop_dominant_policy)
        self.prop_subdominant_policy.append(prop_subdominant_policy)
        self.prop_unreasonable_policy.append(prop_unreasonable_policy)

        prev_prop_dominant_policy = prop_dominant_policy
        prev_prop_subdominant_policy = prop_subdominant_policy
        prev_prop_unreasonable_policy = prop_unreasonable_policy

        
        for iteration in range(n_iterations):

            #Calculate statistics of current behavior map, e.g. what is the dominant policy, how much of the Behavior Map is covered
            #by the dominant policy, how much of the Behavior Map is covered by unreasonable policies.

            #Perturb the reward function.
            self.perturb_reward_function(dominant_states)

            #Perturb the transition function. Only insert wall 30% of the time.
            theta = np.random.uniform(0, 1)
            if theta < 0.3:
                self.perturb_transition_dynamics(dominant_states)

            #Generate new Behavior Map.
            behavior_map_perturbed_environment = self.generate_behavior_map(plot_bmap=plot_bmap)

            #Calculate statistics of new behavior map.
            prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states = self.calculate_behavior_map_stats(
                                                                                                        behavior_map=behavior_map_perturbed_environment)

            if (prop_subdominant_policy - prop_dominant_policy)/prop_unreasonable_policy > (prev_prop_subdominant_policy - prev_prop_dominant_policy)/prev_prop_unreasonable_policy:
                #Accept this environment.

                #Append statistics to list.
                print("Accepted Environment.")
                self.prop_dominant_policy.append(prop_dominant_policy)
                self.prop_subdominant_policy.append(prop_subdominant_policy)
                self.prop_unreasonable_policy.append(prop_unreasonable_policy)

                #Store perturbed environment.
                self.perturbed_environments[f"iteration_{iteration}"] = self.perturbed_environment
                self.perturbed_behavior_maps[f"iteration_{iteration}"] = behavior_map_perturbed_environment

                #Update previous statistics.
                prev_prop_dominant_policy = prop_dominant_policy
                prev_prop_subdominant_policy = prop_subdominant_policy
                prev_prop_unreasonable_policy = prop_unreasonable_policy

                self.n_accepted += 1

            else:
                #Reject this environment. Reset the environment to the previous environment.
                print("Rejected Environment.")
                pass

            if prop_dominant_policy == 0:
                print(f"Terminating perturbation. Dominant policy has been removed.")
                break


    # def generate_behavior_map(self,
    #                           plot_bmap: bool = False):

    #     '''
    #     Calculate the behavior map.
    #     '''

    #     self.world = make_world(height=self.perturbed_environment.N,
    #                       width=self.perturbed_environment.M,
    #                       rewards=self.perturbed_environment.R_true,
    #                       absorbing_states=self.perturbed_environment.goal_states,
    #                       wall_states=self.perturbed_environment.wall_states)

    #     behavior_map_perturbed_environment = bm.plot_bmap(
    #         world=self.world,
    #         gammas=gammas,
    #         probs=probs,
    #         plot=plot_bmap
    #     )

    #     return behavior_map_perturbed_environment

    
    # def plot_behavior_map(self, environment, gammas, probs):

    #     '''
    #     Plot the behavior map of an environment for parameter bounds gammas and probs.
    #     '''

    #     self.world = make_world(height=environment.N,
    #                       width=environment.M,
    #                       rewards=environment.R_true,
    #                       absorbing_states=environment.goal_states,
    #                       wall_states=environment.wall_states)

    #     self.behavior_map_perturbed_environment = bm.plot_bmap(
    #         world=self.world,
    #         gammas=gammas,
    #         probs=probs,
    #         plot=True
    #     )

    def plot_props_over_time(self, iteration=None):

        '''
        Plot the proportion of the Behavior Map covered by the dominant policy, subdominant policy and unreasonable policy over time.

        Args:
        iteration: int, the iteration to plot the proportion of the Behavior Map covered by the dominant policy, subdominant policy and unreasonable policy up to.
        if none: plot all iterations.
        '''

        if iteration is None:
            iteration = len(self.prop_dominant_policy)-1
            
        _iterations = np.arange(0, iteration+1, 1.0)
        plt.plot(_iterations, self.prop_dominant_policy[:iteration+1], "x-", color="green", label="Dominant Policy")
        plt.plot(_iterations, self.prop_subdominant_policy[:iteration+1], "o-", color="red", label="Subdominant Policies")
        plt.plot(_iterations, self.prop_unreasonable_policy[:iteration+1], "s-", color="blue", label="Unreasonable Policy")
        plt.xlabel("Iteration")
        plt.ylabel("Proportion of Behavior Map")
        plt.legend()
        plt.title("Proportion of BM covered by Dominant, Subdominant and Unreasonable Policies.")
        plt.xticks(_iterations)
        plt.show()

        del _iterations