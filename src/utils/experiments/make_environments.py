import numpy as np
from src.utils.make_environment import Environment, transition_matrix, insert_walls_into_T
from src.utils.constants import GenParamTuple





def make_big_small(big_reward, min_p, max_p, min_gamma, max_gamma, resolution, p_true, gamma_true, reward_true, N = 7, M = 7):

    '''
    
    '''


    #Define custom functions to generate reward, transition and gamma.
    def custom_transition_func(p, goal_states, wall_states):

        _T = transition_matrix(N=7, M=7, p=p, absorbing_states=goal_states)
        _T = insert_walls_into_T(T=_T, wall_indices=wall_states)
        return _T

    def custom_gamma_func(gamma):
        return gamma

    def custom_reward_func(big_reward):
        reward_func = np.zeros((N, M))
        reward_func[N-1, 0] = 0.1
        reward_func[N-1, M-1] = big_reward
        return reward_func.flatten()


    #make reward function
    R = np.zeros((N, M))
    R[N-1, 0] = 0.1
    R[N-1, M-1] = big_reward
    R = R.flatten()

    goal_states = np.where(R != 0)[0]

    #make transition function
    wall_states = [14]

    big_small = Environment(
        N=N,
        M=M,
        reward_function = custom_reward_func,
        transition_function=custom_transition_func,
        gamma = custom_gamma_func,
        wall_states=wall_states,
        start_state=0,
        goal_states=goal_states
    )


    #Generate parameter ranges
    resolution = 25
    p_range = np.linspace(min_p, max_p, resolution)
    gamma_range = np.linspace(min_gamma, max_gamma, resolution)
    R_range = np.linspace(0.3, 0.95, resolution)


    gamma_range = gamma_range.reshape(1, resolution)
    p_range = p_range.reshape(1, resolution)
    R_range = R_range.reshape(1, resolution)


    #create true parameters
    p_true = p_true.reshape(1, 1)
    gamma_true = gamma_true.reshape(1, 1)
    reward_true = reward_true.reshape(1, 1)
    true_params = GenParamTuple(T = p_true, gamma=gamma_true, R=reward_true)


    return big_small, p_range, gamma_range, R_range, true_params







