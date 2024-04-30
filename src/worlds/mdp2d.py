from datetime import datetime
from enum import Enum
from typing import Callable, Tuple


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import njit

from src.utils.make_environment import transition_matrix, insert_walls_into_T, transition_matrix_is_valid
from src.utils.optimization import soft_q_iteration
from src.utils.constants import beta_agent



# @njit(nopython=True)
# def value_iteration_with_policy(
#     R: np.ndarray,
#     T_agent: np.ndarray,
#     gamma: float,
#     tol: float = 1e-6,
#     V: np.array = None,
#     policy: np.array = None
# ):
    
#     n_states = R.shape[0]
#     if V is None:
#         V = np.zeros(n_states, dtype=np.float64)
#     if policy is None:
#         policy = np.zeros(n_states, dtype=np.float64)

#     while True:
#         V_new = np.zeros(n_states)
#         for s in range(n_states):
#             action_values = R[s] + gamma * np.sum(T_agent[s] * V, axis=1)
#             best_action = np.argmax(action_values)
#             V_new[s] = action_values[best_action]
#             policy[s] = best_action
#         if np.max(np.abs(V - V_new)) < tol:
#             break
#         V = V_new
#     V = V / np.max(V) * R.max()
#     return V, policy



class MDP_2D:
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma

        self.height = self.S.shape[0]
        self.width = self.S.shape[1]

        self.V = np.zeros(self.S.shape)
        self.policy = np.zeros(self.S.shape)
        self.theta = 0.0001
        self.state = self.S[0][0]

        # Check if transition probabilities are valid
        assert transition_matrix_is_valid(
            T
        ), "The transition probabilities are not proper."


    def make_heatmap(
        self,
        setup_name,
        policy_name,
        labels,
        base_dir="images",
        save=True,
        show=False,
        ax=None,
        mask=None,
    ):
        # draw heatmap and save in figure
        hmap = sns.heatmap(
            self.V,
            annot=labels,
            fmt="",
            xticklabels="",
            yticklabels="",
            cbar=False,
            cbar_kws={"label": "Value"},
            annot_kws={"size": 25 / np.sqrt(len(self.V))},
            ax=ax,
            mask=mask,
        )
        hmap.set(title=f"{policy_name} Value Iteration")
        hmap = hmap.figure
        file_name = policy_name.replace(" ", "_").lower()
        setup_name = setup_name.replace(" ", "_").lower()

        if save:
            filepath = f"{base_dir}/{setup_name}/{file_name}_{datetime.now()}.png"
            print(f"Saving heatmap for {policy_name} in {filepath}")
            plt.savefig(filepath)

        if show:
            plt.show()

    def solve(
        self
    ):

        
        # self.V, self.policy = value_iteration_with_policy(self.R, self.T, self.gamma, V = self.V.flatten(), policy=self.policy.flatten())
        self.policy = soft_q_iteration(self.R, self.T, self.gamma, beta=beta_agent, return_what="policy")

        # self.policy = np.reshape(self.policy,  newshape=(self.height, self.width))
        # self.V = np.reshape(self.V,  newshape=(self.height, self.width))

        #Convert stochastic Boltzmann policy into determinstic, greedy policy for rollouts.
        self.policy = np.argmax(self.policy, axis=1)
        self.policy = np.reshape(self.policy,  newshape=(self.height, self.width))

        return self.policy

    def reset(self):
        self.state = self.S[0][0]


class Experiment_2D:
    def __init__(
        self,
        height: int,
        width: int,
        rewards=None,
        absorbing_states=[],
        wall_states=[],
        action_success_prob=0.8,
        gamma=0.9,
        reward_param_a=0.5,
        reward_param_b=0.5
    ):
        # Assert valid parameters
        assert (
            0 <= action_success_prob <= 1
        ), "Action success probability must be in [0, 1]"
        assert 0 <= gamma <= 1, "Gamma must be in [0, 1]"

        assert (type(height) == int or type(height) == np.int64) and (
            type(width) == int or type(width) == np.int64
        ), "Height and width must be integers"
        assert height > 0 and width > 0, "Height and width must be positive integers"

        self.height = height
        self.width = width
        self.gamma = gamma
        self.rewards = rewards
        self.action_success_prob = action_success_prob
        self.absorbing_states = absorbing_states
        self.wall_states = wall_states

        self.params = None

        S, A, T, R = self.make_MDP_params()
        R = self.rewards
        self.mdp: MDP_2D = MDP_2D(S, A, T, R, gamma)




    def make_MDP_params(self):
        n_states = self.width * self.height
        h, w = self.height, self.width

        S = np.arange(n_states).reshape(h, w)
        A = np.array((0, 1, 2, 3))  # 0 is left, 1 is right, 2 is up, 3 is down

        T = np.zeros((A.shape[0], n_states, n_states))


        T = transition_matrix(N=self.height, M=self.width, p=self.action_success_prob, absorbing_states=self.absorbing_states)
        T = insert_walls_into_T(T=T, wall_indices=self.wall_states)

    

        # #Define Reward Function
        # def cobb_douglas(s, a, b, n_rows, n_cols):
        #     '''
        #     Models Cobb Douglas preferences
        #     '''
        #     row_state = s//n_rows
        #     col_state = s%n_cols
            
        #     return (row_state+1)**a*(col_state+1)**b 

        # R = np.zeros((h,w))
        # R = R.flatten()
        # R = [cobb_douglas(state, a=self.reward_param_a, b=self.reward_param_b, n_rows=h, n_cols=w) for state in np.arange(h*w)]
        # R = np.array(R)
        
        R = self.rewards

        return S, A, T, R


    def solve(self):
        self.mdp.solve()


    def set_user_params(
        self,
        prob: float,
        gamma: float,
        # reward_param_a: float,
        # reward_param_b: float,
        params: dict,
        # transition_func: Callable[..., np.ndarray],
        use_pessimistic: bool = False,
    ) -> MDP_2D:
        """

        """
        self.gamma = gamma
        # self.reward_param_a = reward_param_a
        # self.reward_param_b = reward_param_b

        if not use_pessimistic:
            self.action_success_prob = prob

        S, A, T, R = self.make_MDP_params()
        # T = transition_func(
        #     T=T, height=self.height, width=self.width, prob=prob, params=params
        # )

        self.mdp = MDP_2D(S, A, T, R, self.gamma)

        return self.mdp
