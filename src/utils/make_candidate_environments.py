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
                 region_of_interest,
                 function_init,
                 verbose,
                 learn_what,
                 wall_states) -> None:

        self.estimate_R = estimate_R
        self.estimate_gamma = estimate_gamma
        self.estimate_T = estimate_T

        self.named_parameter_mesh = named_parameter_mesh
        self.shaped_parameter_mesh = shaped_parameter_mesh
        self.region_of_interest = region_of_interest
        self.verbose = verbose
        self.function_init = function_init
        self.learn_what = learn_what
        self.wall_states = wall_states


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
    
    def masked_softmax(self, T_tilde, inv_temperature = 100):

        '''
        TODO
        '''
        
        # Create a mask where T_tilde is zero
        mask = (T_tilde != 0).float()

        # Multiply by inverse temperature to have sharper max
        T_tilde = T_tilde * inv_temperature
        
        # Apply softmax, but only to non-zero elements
        # Subtract the max for numerical stability
        T_tilde_max = torch.max(T_tilde * mask, dim=-1, keepdim=True)[0]
        exp_T_tilde = torch.exp((T_tilde - T_tilde_max) * mask) * mask
        
        # Normalize by the sum of the exponentials
        sum_exp_T_tilde = torch.sum(exp_T_tilde, dim=-1, keepdim=True)
        T_softmax = exp_T_tilde / sum_exp_T_tilde
        
        return T_softmax
    
    def min_max_normalize(self, T_tilde):
        
        # Create a mask where T_tilde is zero
        mask = (T_tilde != 0).float()
        
        # Compute the min and max values along the last dimension (states dimension)
        T_tilde_min = torch.min(T_tilde * mask + (1 - mask) * float('inf'), dim=-1, keepdim=True)[0]
        T_tilde_max = torch.max(T_tilde * mask + (1 - mask) * float('-inf'), dim=-1, keepdim=True)[0]
        
        # Apply min-max normalization
        T_normalized = (T_tilde - T_tilde_min) / (T_tilde_max - T_tilde_min + 1e-10) * mask
        
        return T_normalized


    def update_covers(self, function_init, bm_out, update_how, how_args, update_what = "R"):

        '''
        Update a reward function R to maximize the entropy of the behavior map.

        Args:
        - R_init (np.array): The initial reward function.
        - bm_out (dict): The output of the behavior map function.
        - Update_how ['gradient', 'random']: How to update the reward function. Gradient responds to AMBER.
        - how_args (dict): Arguments for the how method.

        Returns:
        - R (numpy.tensor): The updated reward function R.
        '''

        covers, max_ent_cover = self.compute_covers(bm_out)

        # Initialize learning parameters
        if update_how == "gradient":
            R = torch.tensor(self.estimate_R, dtype=torch.float32)
            gamma = torch.tensor(self.estimate_gamma, dtype=torch.float32)
            T = torch.tensor(self.estimate_T, dtype=torch.float32)
            V_star = torch.zeros_like(R)

        elif update_how == "random":
            if update_what == "R":
                R = function_init
            else:
                T = function_init

        if update_how =="gradient":
            if update_what == "R":
                R = torch.tensor(function_init, dtype=torch.float32)
            elif update_what == "T":
                T = torch.tensor(function_init, dtype=torch.float32)

        for _ in range(how_args["n_iterations"]):

            if update_how == "gradient":
                # Compute the gradient of the value function with respect to the reward function and the transition matrix.
                V_star, R_grad_out, T_grad_out = differentiate_V(R = R, gamma = gamma, T = T, V = V_star)

            # print("initial T: ", T)
            # print("Gradient T: ", T_grad_out)
            # Update the reward function
            for behavior_idx in covers:

                #Get states that are visited by the behavior.
                cover = covers[behavior_idx]
                _visited_states = bm_out.pidx2states[behavior_idx]

                if update_how == "gradient":
                    
                    #Get gradient along those visited states.
                    if update_what == "R":
                        _masked_gradient_R = torch.zeros_like(R_grad_out)
                        _masked_gradient_R[_visited_states] = R_grad_out[_visited_states]

                    elif update_what == "T":
                        _masked_gradient_T = torch.zeros_like(T_grad_out)
                        _masked_gradient_T[_visited_states,:,:] = T_grad_out[_visited_states,:,:]


                    #Inhibit behavior that covers more than maximum entropy share.
                    if (cover > max_ent_cover) or (cover == 1):

                        if update_what == "R":
                            R = R - how_args["stepsize"] * _masked_gradient_R
                        elif update_what == "T":
                            T = T + how_args["stepsize"] * _masked_gradient_T
                            # pass


                    #Excite behavior that covers less than maximum entropy share.
                    else:
                        if update_what == "R":
                            R = R + how_args["stepsize"] * _masked_gradient_R
                        elif update_what == "T":
                            # T = T - how_args["stepsize"] * _masked_gradient_T
                            pass

                elif update_how == "random":

                    if (cover > max_ent_cover) or (cover == 1):

                        if update_what == "R":
                            R = self.change_function_randomly(states = _visited_states, update_what = update_what, what = "inhibit", initial_function = R, stepsize = how_args["stepsize"])
                        elif update_what == "T":
                            T = self.change_function_randomly(states = _visited_states, update_what = update_what, what = "inhibit", initial_function = T, stepsize = how_args["stepsize"])
                    
                    else:
                        if update_what =="R":
                            R = self.change_function_randomly(states = _visited_states, what = "excite", initial_function = R, stepsize = how_args["stepsize"])

            if update_how == "gradient":
                R = R.detach().numpy()


            # print("Update T: ", T)
            # print("Normalized T:", T/torch.sum(T, axis=2, keepdim=True))
            # print("Update T softmax: ", self.masked_softmax(T, inv_temperature=2.5))
            # print("Normalized T:", self.min_max_normalize(T))


        if update_what == "R":
            return R
        elif update_what == "T":
            #Normalize T via softmax.
            # return self.min_max_normalize(T)
            # return self.masked_softmax(T, inv_temperature=2.5)

            if update_how == "gradient":
                T_norm = T / torch.sum(T, axis=2, keepdim=True)
                T_norm = insert_walls_into_T(T=T_norm.detach().numpy(), wall_indices=self.wall_states)
                return T_norm
            elif update_how == "random":
                return T


    def change_function_randomly(self, states, update_what, what: str, initial_function: np.array, stepsize: float):

        '''
        Randomly change reward function along rollout of policy.

        Args:
        - states (list): The states along which the reward function is changed.
        - what (str): What to do with the reward function. Either 'excite' or 'inhibit'.
        - reward_function (np.array): The reward function to change.

        Returns:
        - reward_function (np.array): The updated reward function.
        '''


        assert what in ["excite", "inhibit"], "What to do with the reward function not recognized. Choose either 'excite' or 'inhibit'."

        #Remove start state and goal states from states.
        admissable_states = np.setdiff1d(list(states), [0]) #TODO infer this, currently hard coded.
        # admissable_states = np.setdiff1d(admissable_states, self.environment.goal_states)

        if len(admissable_states) == 0:
            return initial_function
        
        random_state = np.random.choice(admissable_states, size = 1)

        if update_what == "R":
            random_reward_ranges_min = 0
            random_reward_ranges_max = 0.5*np.max(initial_function)


            if what == "excite":
                initial_function[random_state] += stepsize*np.random.uniform(low = random_reward_ranges_min, high = random_reward_ranges_max)
            elif what == "inhibit":
                initial_function[random_state] -= stepsize*np.random.uniform(low = random_reward_ranges_min, high = random_reward_ranges_max)

            return initial_function
        
        elif update_what == "T": #transition function

            if what == "inhibit":
                if isinstance(initial_function, torch.Tensor):
                    initial_function = initial_function.detach().numpy()
                T = insert_walls_into_T(T=initial_function, wall_indices=random_state)
                return torch.tensor(T, dtype=torch.float32)

    
    
    def BM_search(self, 
                  base_environment, 
                  named_parameter_mesh, 
                  candidate_environment_args,
                  search_how: str = "gradient",
                ):
        '''
        Find a reward function that maximizes the entropy of the Behavior Map.

        Args:
        TODO    
        '''

        assert search_how in ["gradient", "random"], "Search method not recognized. Choose either 'gradient' or 'random'."

        environment = deepcopy(base_environment)
        # if self.reward_init is None:
        #     R = self.estimate_R
        # else:
        #     R = self.reward_init

        _max_ent = -np.inf
        _max_ent_cover = None
        # max_ent_R = self.estimate_R
        if "R" in self.learn_what:
            entropy_update = self.estimate_T
        elif "T" in self.learn_what:
            entropy_update = self.estimate_R

        # entropy_update = np.zeros_like(R)
        region_of_interest = self.region_of_interest

        entropy_maximized: bool = False
        n_iterations = 0

        #Set up diagnostics tracking.
        diags = {}
        diags["diagnostics_cover_numbers"] = []
        diags["diagnostics_entropy"] = []
        diags["diagnostics_entropy_BM_last_iteration"] = None

        verbose_BM: bool = False

        # for _ in range(n_compute_BM):
        while not entropy_maximized:


            # Compute Behavior Map
            bm_out = bm.calculate_behavior_map(environment=environment,
                                               entropy_update=entropy_update,
                                               parameter_mesh=named_parameter_mesh,
                                               region_of_interest = region_of_interest,
                                               learn_what = self.learn_what,
                                               verbose_BM = verbose_BM)
            
            print("Behavior Map:", bm_out)

            #Compute entropy of BM
            cover, max_ent_prob = self.compute_covers(bm_out)
            entropy_BM = stats.entropy(list(cover.values()))

            print("Cover: ", cover)

            #Check if the current Behavior Map has higher entropy.
            if entropy_BM > _max_ent:
                max_ent_possible = stats.entropy(np.repeat(max_ent_prob, repeats=int(1/max_ent_prob)))
                _max_ent = entropy_BM
                _max_ent_cover = cover
                max_ent_update = entropy_update
                _max_ent_BM = bm_out

            # Perform Gradient Updates on Reward Function to maximize entropy of BM.
            if search_how == "gradient":

                if "R" in self.learn_what:
                    entropy_update = self.update_covers(function_init = entropy_update, bm_out=bm_out, update_how = "gradient", how_args = candidate_environment_args, update_what = "T")
                elif "T" in self.learn_what:
                    entropy_update = self.update_covers(function_init = entropy_update, bm_out=bm_out, update_how = "gradient", how_args = candidate_environment_args, update_what = "R")



            # Randomly change reward function along rollout of policy.
            elif search_how == "random":
                if "R" in self.learn_what:
                    entropy_update = self.update_covers(function_init = entropy_update, bm_out=bm_out, update_how = "random", how_args = candidate_environment_args, update_what = "T")
                elif "T" in self.learn_what:
                    entropy_update = self.update_covers(function_init = entropy_update, bm_out=bm_out, update_how = "random", how_args = candidate_environment_args, update_what = "R")

            #Check if the entropy of the Behavior Map has been maximized.
            if (np.isclose(_max_ent, max_ent_possible, rtol = 0.01)) and max_ent_possible != 0:
                entropy_maximized = True

            n_iterations += 1

            if n_iterations > candidate_environment_args["n_compute_BM"]:
                print("\n\nReached Maximum Number of BM Computations. Terminating BM Search.\n\n")
                break

            # print("Iteration: ", n_iterations)
            # print("Reward Function: ", R)

            
            #Append diagnostics.
            diags["diagnostics_cover_numbers"].append(cover)
            diags["diagnostics_entropy"].append(entropy_BM)
            diags["diagnostics_entropy_BM_last_iteration"] = _max_ent

            # verbose_BM = True


        if self.verbose:
            print(f"Finished BM Search. Entropy: {_max_ent}. Max Ent possible: {max_ent_possible}. Cover: {_max_ent_cover}. Behaviors: {bm_out.pidx2states}")
            print("Behavior map: ", _max_ent_BM)
            print("Entropy Function: ", max_ent_update)


        return max_ent_update, diags
        


# class AgnosticsBM():

#     '''
#     Calculate the Behavior Map of an environment and perturb the environment such that the dominant policy becomes less attractive 
#     and the subdominant policies become more attractive.
#     '''

#     def __init__(self, 
#                  environment,
#                  behavior_map_environment,
#                  region_of_interest=None):
        
#         self.environment = environment
#         self.perturbed_environment = deepcopy(self.environment)
#         self.region_of_interest  = region_of_interest
#         self.behavior_map_base_environment = behavior_map_environment
#         self.behavior_map_perturbed_environment = None
#         self.prop_dominant_policy = []
#         self.prop_subdominant_policy = []
#         self.prop_unreasonable_policy = []
#         self.perturbed_environments = {}
#         self.perturbed_behavior_maps = {}
#         self.n_accepted = 0

#     def calculate_behavior_map_stats(self, behavior_map):

#         '''
#         Determine reasonable, unreasonable and dominant policies in the behavior map.
#         '''

#         #Reasonable policies are all policies that don't stay in the start state. Unreasonable policies stay in the start state.
#         reasonable_policies_idx = [p for p, states in behavior_map.pidx2states.items() if states[-1] in self.environment.goal_states]
#         unreasonable_policies_idx = [p for p, states in behavior_map.pidx2states.items() if states[-1] not in self.environment.goal_states]

#         #Number of different policies.
#         behaviors_flattened = behavior_map.data.flatten()
#         #Only get behaviors in Region of interest.
#         if self.region_of_interest is not None:
            
#             behaviors_flattened = behavior_map.data[tuple(self.region_of_interest.T)]
#         n_behaviors = len(behaviors_flattened)

#         if self.region_of_interest is not None:
#             assert n_behaviors == self.region_of_interest.shape[0], "The number of behaviors in the region of interest does not match the number of behaviors in the behavior map."

#         #Number of reasonable policies.
#         n_reasonable_behaviors = np.sum(np.isin(behaviors_flattened, reasonable_policies_idx))
#         n_unreasonable_behaviors = np.sum(np.isin(behaviors_flattened, unreasonable_policies_idx))

#         #Count the number of times each policy is chosen.
#         behavior, counts = np.unique(behaviors_flattened, return_counts=True)
#         behavior_counts = dict(zip(behavior, counts))

#         #Dominant policy is the policy with the most counts, e.g. that covers the largest proportion of the Behavior Map.
#         # Remove all unreasonable policies from the counts.
#         for unreasonable_policy in unreasonable_policies_idx:
#             behavior_counts.pop(unreasonable_policy, None)

#         #Get the dominant policy. If there are only unreasonable policies (e.g. behavior counts ie empty), then the dominant policy is None.
#         if behavior_counts == {}:
#             dominant_policy = None
#             return 0, 0, 1, None
#         else:
#             dominant_policy = max(behavior_counts, key=behavior_counts.get)

#         #Subdominant policies are all policies that are a) not dominant and b) not unreasonable.
#         n_dominant_behaviors = behavior_counts[dominant_policy]
#         n_subdominant_behaviors = n_reasonable_behaviors - n_dominant_behaviors

#         #Determine proportion of BM that each policy type covers.
#         prop_dominant_policy = n_dominant_behaviors / n_behaviors
#         prop_subdominant_policy = n_subdominant_behaviors / n_behaviors
#         prop_unreasonable_policy = n_unreasonable_behaviors / n_behaviors

#         #Get a rollout of the dominant policy.
#         dominant_states = behavior_map.pidx2states[dominant_policy]

#         return prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states

#     def perturb_transition_dynamics(self, states):

#         '''
#         We want to change the transition dynamics, such that prop_dominant_policy decreases and prop_subdominant_policy increases while prop
#         _unreasonable_policy remains small.
#         We can do this by changing the reward function and transition dynamics such that the dominant policy becomes less attractive and the subdominant
#         poicies become more attractive. To this end, we insert walls/ death states along the rollouts of the dominant policy and remove walls/death states
#         from the rollouts of the subdominant policies. This will make the dominant policy less attractive and the subdominant policies more attractive.
#         '''
#         #Get a random state from the dominant policy to insert a wall into. Remove goal states and start state.
#         random_state_from_dominant = np.random.choice(list(states), size = 1)
#         random_state_from_dominant = np.setdiff1d(random_state_from_dominant, self.environment.goal_states)
#         #TODO: start state should be flexible, not 0.
#         random_state_from_dominant = np.setdiff1d(random_state_from_dominant, [0])

#         #Insert walls into the transition matrix along the rollouts of the dominant policy and update transition function.
#         T_new = insert_walls_into_T(T=self.perturbed_environment.T_true, wall_indices=random_state_from_dominant)
#         self.perturbed_environment.wall_states = np.append(self.perturbed_environment.wall_states, random_state_from_dominant)
#         self.perturbed_environment.T_true = T_new
#         print(f"Perturbed transition dynamics. Inserted a wall into state {random_state_from_dominant}.")

    
#     def perturb_reward_function(self, states):

#         '''
#         Change reward function along dominant rollout of dominant.
#         '''
#         #Get a random state from the dominant policy to insert a wall into. Remove goal states and start state.
#         random_state_from_dominant = np.random.choice(list(states), size = 1)
#         random_state_from_dominant = np.setdiff1d(random_state_from_dominant, self.environment.goal_states)
#         #TODO: start state should be flexible, not 0.
#         random_state_from_dominant = np.setdiff1d(random_state_from_dominant, [0])
#         #Insert negative reward into the transition matrix along the rollouts of the dominant policy.
#         R_new = self.perturbed_environment.R_true.copy()
#         R_new[random_state_from_dominant] += -0.1

#         self.perturbed_environment.R_true = R_new
#         print(f"Perturbed reward function. Inserted a negative reward into state {random_state_from_dominant}.")

    
#     def perturb_environment(self, n_iterations: int,
#                             plot_bmap: bool = False,):

#         '''
#         Perturb the environment such that the dominant policy becomes less attractive and the subdominant policies become more attractive.
#         '''

#         #Statistics of Base Environment.
#         prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states = self.calculate_behavior_map_stats(behavior_map=self.behavior_map_base_environment)
#         self.prop_dominant_policy.append(prop_dominant_policy)
#         self.prop_subdominant_policy.append(prop_subdominant_policy)
#         self.prop_unreasonable_policy.append(prop_unreasonable_policy)

#         prev_prop_dominant_policy = prop_dominant_policy
#         prev_prop_subdominant_policy = prop_subdominant_policy
#         prev_prop_unreasonable_policy = prop_unreasonable_policy

        
#         for iteration in range(n_iterations):

#             #Calculate statistics of current behavior map, e.g. what is the dominant policy, how much of the Behavior Map is covered
#             #by the dominant policy, how much of the Behavior Map is covered by unreasonable policies.

#             #Perturb the reward function.
#             self.perturb_reward_function(dominant_states)

#             #Perturb the transition function. Only insert wall 30% of the time.
#             theta = np.random.uniform(0, 1)
#             if theta < 0.3:
#                 self.perturb_transition_dynamics(dominant_states)

#             #Generate new Behavior Map.
#             behavior_map_perturbed_environment = self.generate_behavior_map(plot_bmap=plot_bmap)

#             #Calculate statistics of new behavior map.
#             prop_dominant_policy, prop_subdominant_policy, prop_unreasonable_policy, dominant_states = self.calculate_behavior_map_stats(
#                                                                                                         behavior_map=behavior_map_perturbed_environment)

#             if (prop_subdominant_policy - prop_dominant_policy)/prop_unreasonable_policy > (prev_prop_subdominant_policy - prev_prop_dominant_policy)/prev_prop_unreasonable_policy:
#                 #Accept this environment.

#                 #Append statistics to list.
#                 print("Accepted Environment.")
#                 self.prop_dominant_policy.append(prop_dominant_policy)
#                 self.prop_subdominant_policy.append(prop_subdominant_policy)
#                 self.prop_unreasonable_policy.append(prop_unreasonable_policy)

#                 #Store perturbed environment.
#                 self.perturbed_environments[f"iteration_{iteration}"] = self.perturbed_environment
#                 self.perturbed_behavior_maps[f"iteration_{iteration}"] = behavior_map_perturbed_environment

#                 #Update previous statistics.
#                 prev_prop_dominant_policy = prop_dominant_policy
#                 prev_prop_subdominant_policy = prop_subdominant_policy
#                 prev_prop_unreasonable_policy = prop_unreasonable_policy

#                 self.n_accepted += 1

#             else:
#                 #Reject this environment. Reset the environment to the previous environment.
#                 print("Rejected Environment.")
#                 pass

#             if prop_dominant_policy == 0:
#                 print(f"Terminating perturbation. Dominant policy has been removed.")
#                 break


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