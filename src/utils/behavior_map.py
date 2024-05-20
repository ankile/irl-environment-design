from collections import namedtuple
from typing import Dict

import numpy as np

from src.utils.policy import follow_policy
from src.utils.optimization import soft_q_iteration
from src.utils.constants import beta_agent
from src.utils.make_environment import Environment


ExperimentResult = namedtuple("ExperimentResult", ["data", "p2idx", "pidx2states"])



#TODO, only compute over Region of Interest.
def calculate_behavior_map(
    environment: Environment,
    reward_update: np.ndarray,
    parameter_mesh,
    shaped_parameter_mesh,
    region_of_interest: np.ndarray,
) -> ExperimentResult:
    """
    Run an experiment with a given set of parameters and return the results.

    The results are:
    - data: a matrix of shape (len(probs), len(gammas)) where each entry is an index
            into the list of policies
    - p2idx: a dictionary mapping policies to indices
    - pidx2states: a dictionary mapping indices to states visited by the policy
    """

    data: np.ndarray = np.zeros_like(region_of_interest, dtype=np.int32)
    p2idx: Dict[str, int] = {}
    pidx2states: Dict[list, int] = {}


    #Index for current policy, increased by 1 for each new policy.
    idx_policy = 0

    #Initialize policy, V, Q
    policy = None
    V = None
    Q = None
    idx_ROI = 0

    for idx_parameter, parameter in enumerate(parameter_mesh):


        #Compute BM restricted to ROI.
        if idx_parameter in region_of_interest:
        
            #Get the transition function, reward function, and gamma from the parameter.
            _transition_func = environment.transition_function(*parameter.T)
            _reward_func = environment.reward_function(*parameter.R)
            _gamma = parameter.gamma


            #Update the reward function with the maximum entropy reward update from the previous iteration.
            _reward_func += reward_update

            #Run soft Q-iteration to get the optimal policy.
            policy, Q, V = soft_q_iteration(
                _reward_func, _transition_func, gamma=_gamma, beta=beta_agent, return_what="all", Q_init=Q, V_init=V, policy_init=policy
            )

            #Convert stochastic Boltzmann policy into determinstic, greedy policy for rollouts.
            greedy_policy = np.argmax(policy, axis=1)
            greedy_policy = np.reshape(greedy_policy,  newshape=(environment.N, environment.M))


            policy_str, policy_states = follow_policy(
                greedy_policy,
                height=environment.N,
                width=environment.M,
                initial_state=environment.start_state,
                goal_states=environment.goal_states,
            )

            equivalent_policy_exists: bool = False
            
            if pidx2states == {}:
                #First iteration, no equivalent policies yet.
                p2idx[policy_str] = idx_policy
                pidx2states[idx_policy] = policy_states
                idx_policy += 1


            else:
            
                #Get all previous rollouts/ policies.
                policy_rollouts = pidx2states.values()

                #We initialize the equivalent policy as the current policy. If there exists an equivalent one, we later overwrite it.
                for policy_rollout in policy_rollouts:

                    #Check if there exists an equivalent policy already. Here, we define equivalent as
                    # two policies are equivalent if their rollouts are equal up to a permutation (in previous
                    # versions, we defined two policies to be only equivalent if their rollouts are exactly the same).
                    # We can test equality up to a permutation more efficiently by testing whether the policies have
                    #the same length and whether the policies arrive in the same goal state.

                    if (len(policy_rollout) == len(policy_states)) and (policy_rollout[-1] == policy_states[-1]):
                        # Check whether there exists an equivalent policy (up to permutation).
                        equivalent_policy_exists = True
                        equivalent_policy_rollout = policy_rollout

                        #Get index of equivalent policy.
                        equivalent_policy_rollout_idx = list(pidx2states.keys())[list(pidx2states.values()).index(equivalent_policy_rollout)]
                        break


                if not equivalent_policy_exists:
                    #There exists no equivalent policy, so new policy index is created
                    p2idx[policy_str] = idx_policy
                    pidx2states[idx_policy] = policy_states
                    idx_policy += 1

            #Update which policy sample was used.
            if equivalent_policy_exists:
                data[idx_ROI] = equivalent_policy_rollout_idx

            else:
                data[idx_ROI] = p2idx[policy_str]

            idx_ROI += 1

    return ExperimentResult(data, p2idx, pidx2states)