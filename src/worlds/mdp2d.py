from datetime import datetime
from enum import Enum
from typing import Callable, Tuple


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import njit


from src.utils.enums import TransitionMode
from src.utils.transition_matrix import make_absorbing, transition_matrix_is_valid
from src.utils.make_environment import transition_matrix, insert_walls_into_T

# from ..utils import optimization
# from src.utils.optimization import value_iteration_with_policy



# @njit
# def bellman_eq(
#     A: np.ndarray,
#     V: np.ndarray,
#     R: np.ndarray,
#     T: np.ndarray,
#     gamma: float,
#     width: int,
#     row: int,
#     col: int,
# ) -> np.ndarray:
#     state = row * width + col
#     vals = np.zeros(len(A))

#     for action in A:
#         transition_probs = T[action][state]
#         rewards = R[state][action]
#         vals[action] = np.sum(transition_probs * (rewards + gamma * V.flatten()))

#     return vals


# @njit
# def value_iteration(
#     V: np.ndarray,
#     policy: np.ndarray,
#     S: np.ndarray,
#     A: np.ndarray,
#     T: np.ndarray,
#     R: np.ndarray,
#     gamma: float,
#     theta: float,
#     width: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     difference = np.inf
#     while difference >= theta:
#         difference = 0
#         for state in S.flatten():
#             row, col = state // width, state % width
#             old_V = V[row, col]
#             v = bellman_eq(A, V, R, T, gamma, width, row, col)

#             policy[row, col] = np.argmax(v)
#             V[row, col] = np.max(v)

#             difference = np.maximum(difference, np.abs(old_V - V[row, col]))

#     return V, policy


# @jit(nopython=True)
def value_iteration_with_policy(
    R: np.ndarray,
    T_agent: np.ndarray,
    gamma: float,
    tol: float = 1e-6,
):
    n_states = R.shape[0]
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=np.int32)
    while True:
        V_new = np.zeros(n_states)
        for s in range(n_states):
            action_values = R[s] + gamma * np.sum(T_agent[s] * V, axis=1)
            best_action = np.argmax(action_values)
            V_new[s] = action_values[best_action]
            policy[s] = best_action
        if np.max(np.abs(V - V_new)) < tol:
            break
        V = V_new
    V = V / np.max(V) * R.max()
    return V, policy


# @njit
def soft_q_iteration(
    R: np.ndarray,  # R is a one-dimensional array with shape (n_states,)
    T_agent: np.ndarray,
    gamma: float,
    beta: float,  # Inverse temperature parameter for the softmax function
    tol: float = 1e-6,
) -> np.ndarray:
    n_states, n_actions, _ = T_agent.shape
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    policy = np.zeros((n_states, n_actions))

    while True:
        for s in range(n_states):
            for a in range(n_actions):
                # Calculate the Q-value for action a in state s
                Q[s, a] = R[s] + gamma * np.dot(T_agent[s, a], V)

        # Apply softmax to get a probabilistic policy
        max_Q = np.max(Q, axis=1, keepdims=True)
        # Subtract max_Q for numerical stability
        exp_Q = np.exp(beta * (Q - max_Q))
        policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        # Calculate the value function V using the probabilistic policy
        V_new = np.sum(policy * Q, axis=1)
        # V_new = sum_along_axis_1(policy * Q)

        # Check for convergence
        if np.max(np.abs(V - V_new)) < tol:
            break

        V = V_new

    return policy


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

        # # sanity checks:
        # assert T.shape == (
        #     len(self.A),
        #     self.height * self.width,
        #     self.height * self.width,
        # )  # action x state x state

        # Check if transition probabilities are valid
        assert transition_matrix_is_valid(
            T
        ), "The transition probabilities are not proper."

        # assert R.shape == (
        #     self.height * self.width,
        #     len(self.A),
        #     self.height * self.width,
        # )  # state x action x next_state

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
        self,
        setup_name="Placeholder Setup Name",
        policy_name="Placeholder Policy Name",
        save_heatmap=True,
        show_heatmap=False,
        heatmap_ax=None,
        heatmap_mask=None,
        base_dir="images",
        label_precision=3,
    ):
        # Run value iteration
        # self.V, self.policy = value_iteration(
        #     self.V,
        #     self.policy,
        #     self.S,
        #     self.A,
        #     self.T,
        #     self.R,
        #     self.gamma,
        #     self.theta,
        #     self.width,
        # )
        # self.V, self.policy = value_iteration_with_policy(self.R, self.T, self.gamma)
        self.policy = 
        precision = label_precision

        arrows = ["\u2190", "\u2192", "\u2191", "\u2193"]

        self.policy = np.reshape(self.policy,  newshape=(self.height, self.width))
        self.V = np.reshape(self.V,  newshape=(self.height, self.width))

        if len(self.policy) > 0:
            grid = []
            for row in self.S:
                grid_row = []
                for state in row:
                    row, col = state // self.width, state % self.width
                    policy = self.policy[row][col].astype(int)
                    value = self.V[row][col]
                    grid_row.append(f"{arrows[policy]}\n{value:.{precision}f}")
                grid.append(grid_row)

            labels = np.array(grid)

            if save_heatmap or show_heatmap or heatmap_ax:
                self.make_heatmap(
                    setup_name,
                    policy_name,
                    labels,
                    base_dir=base_dir,
                    save=save_heatmap,
                    show=show_heatmap,
                    ax=heatmap_ax,
                    mask=heatmap_mask,
                )

        return self.V, self.policy

    def reset(self):
        self.state = self.S[0][0]


class Experiment_2D:
    def __init__(
        self,
        height: int,
        width: int,
        absorbing_states=[],
        wall_states=[],
        action_success_prob=0.8,
        rewards_dict={-1: 100, -2: -100, -6: -100, -10: -100},
        gamma=0.9,
        transition_mode: TransitionMode = TransitionMode.FULL,
    ):
        # Assert valid parameters
        assert (
            0 <= action_success_prob <= 1
        ), "Action success probability must be in [0, 1]"
        assert 0 <= gamma <= 1, "Gamma must be in [0, 1]"
        # assert (
        #     transition_mode in TransitionMode
        # ), f"Transition mode must be one of {TransitionMode}"
        assert (
            0 <= len(rewards_dict) <= height * width
        ), "Number of rewards must be in [0, height * width]"
        assert (type(height) == int or type(height) == np.int64) and (
            type(width) == int or type(width) == np.int64
        ), "Height and width must be integers"
        assert height > 0 and width > 0, "Height and width must be positive integers"

        self.height = height
        self.width = width
        self.gamma = gamma
        self.action_success_prob = action_success_prob
        self.absorbing_states = absorbing_states
        self.wall_states = wall_states
        self.transition_mode = transition_mode

        self.rewards_dict = self.fix_rewards_dict(rewards_dict)

        self.params = None

        S, A, T, R = self.make_MDP_params()
        self.mdp: MDP_2D = MDP_2D(S, A, T, R, gamma)

    def fix_rewards_dict(self, rewards_dict):
        fixed_rewards_dict = {}
        for idx in rewards_dict:
            if not (idx >= 0 and idx < self.width * self.height):
                fixed_rewards_dict[idx % (self.width * self.height)] = rewards_dict[idx]
            else:
                fixed_rewards_dict[idx] = rewards_dict[idx]
        rewards_dict = fixed_rewards_dict
        return rewards_dict

    @staticmethod
    def _get_target(i, action, width, height):
        row, col = i // width, i % width
        left, right, up, down = i - 1, i + 1, i - width, i + width

        if action == 0:  # left
            target = left if col > 0 else i
        elif action == 1:  # right
            target = right if col < width - 1 else i
        elif action == 2:  # up
            target = up if row > 0 else i
        else:  # down
            target = down if row < height - 1 else i

        return target

    # @staticmethod
    # def _fill_transition_matrix(
    #     T,
    #     A,
    #     height,
    #     width,
    #     action_success_prob,
    #     absorbing_states: np.ndarray=None,
    #     wall_states: np.ndarray=None, 
    #     mode: TransitionMode = TransitionMode.FULL,
    # ) -> None:
        
    #     def _set_probs_for_state_simple(i, action, target):
    #         if target == i:
    #             T[action, i, i] = 1
    #         else:
    #             T[action, i, target] = action_success_prob
    #             T[action, i, i] = 1 - action_success_prob

    #     def _set_probs_for_state(i, action, target):
    #         def in_bounds(row, col):
    #             return 0 <= row < height and 0 <= col < width

    #         row, col = i // width, i % width

    #         # Update transition probability for intended action
    #         # Target could end up in same state if action would take agent out of bounds
    #         T[action, i, target] = action_success_prob

    #         # Calculate remaining probability
    #         remaining_prob = (1 - action_success_prob) / 4

    #         # Update transition probabilities for neighbors
    #         for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
    #             dr, dc = d
    #             r, c = row + dr, col + dc
    #             if in_bounds(r, c) and (neighbor := r * width + c) != target:
    #                 T[action, i, neighbor] = remaining_prob
    #             else:
    #                 T[action, i, i] += remaining_prob

    #     set_functions = {
    #         # TransitionMode.SIMPLE: _set_probs_for_state_simple,   
    #         TransitionMode.FULL: _set_probs_for_state,
    #     }

    #     assert mode in set_functions, f"Mode {mode} not supported"
    #     set_fun = set_functions[mode]
    #     set_fun = _set_probs_for_state

    #     for action in A:
    #         for i in range(width * height):
    #             # Determine the intended target
    #             target = Experiment_2D._get_target(i, action, width, height)
    #             set_fun(i, action, target)


    def make_MDP_params(self):
        n_states = self.width * self.height
        h, w = self.height, self.width

        S = np.arange(n_states).reshape(h, w)
        A = np.array((0, 1, 2, 3))  # 0 is left, 1 is right, 2 is up, 3 is down

        T = np.zeros((A.shape[0], n_states, n_states))

        # Experiment_2D._fill_transition_matrix(
        #     T=T,
        #     A=A,
        #     height=h,
        #     width=w,
        #     action_success_prob=self.action_success_prob,
        #     mode=self.transition_mode,
        # )

        T = transition_matrix(N=self.height, M=self.width, p=self.action_success_prob, absorbing_states=self.absorbing_states)
        T = insert_walls_into_T(T=T, wall_indices=self.wall_states)

        #old codebase had transition matrix as (states, actions, actions), this codebase uses
        # (actions, states, states)
        # T = np.transpose(T, axes=(1,0,2))
        

        # # Define a helper function to assign rewards from the rewards dictionary
        # def assign_reward(idx, magnitude):
        #     # check right border
        #     if idx + 1 % w != w and idx + 1 < n_states:
        #         R[idx + 1, 0, idx] = magnitude
        #     # check left border
        #     if idx - 1 % w != w - 1 and idx - 1 >= 0:
        #         R[idx - 1, 1, idx] = magnitude
        #     # check bottom border
        #     if idx <= w * (h - 1) and idx + w < n_states:
        #         R[idx + w, 2, idx] = magnitude
        #     # check top border
        #     if idx >= w and idx - w >= 0:
        #         R[idx - w, 3, idx] = magnitude

            # Add reward to the state itself
            # R[idx, :, idx] = magnitude

        # previous state, action, new state
        # R = np.zeros((n_states, 4, n_states))
        R = np.zeros(n_states)
        
        for idx in self.rewards_dict:
            R[idx] = self.rewards_dict[idx]
        # # Make reward states absorbing and assign rewards
        # for idx in self.rewards_dict:
        #     if self.rewards_dict[idx] > 0:
        #         make_absorbing(T, idx)

        #     assign_reward(idx, self.rewards_dict[idx])

        return S, A, T, R


    def solve(self):
        self.mdp.solve()

    def myopic(self, gamma):
        self.mdp.gamma = gamma

    def confident(self, action_success_prob):
        # probability is LOWER than the "true": UNDERCONFIDENT
        self.action_success_prob = action_success_prob
        S, A, T, R = self.make_MDP_params()
        self.mdp = MDP_2D(S, A, T, R, self.gamma)

    def pessimistic(
        self,
        scaling,
        new_gamma=None,
        transition_mode: TransitionMode = TransitionMode.SIMPLE,
    ):
        self.transition_mode = transition_mode
        S, A, T, R = self.make_MDP_params()

        # Change the transition probabilities to be more pessimistic
        neg_rew_idx = [idx for idx in self.rewards_dict if self.rewards_dict[idx] < 0]

        T[:, :, neg_rew_idx] *= scaling
        T /= T.sum(axis=2, keepdims=True)

        if new_gamma is not None:
            self.gamma = new_gamma

        self.mdp = MDP_2D(S, A, T, R, self.gamma)

    def set_user_params(
        self,
        prob: float,
        gamma: float,
        params: dict,
        transition_func: Callable[..., np.ndarray],
        use_pessimistic: bool = False,
    ) -> MDP_2D:
        """
        For now, prob can serve as both:
        - 1. The normal notion of probability of success
        - 2. The scaling factor for pessimistic transitions

        (Hopefully, better ways to do this will be implemented in the future)
        """
        self.gamma = gamma

        if not use_pessimistic:
            self.action_success_prob = prob

        S, A, T, R = self.make_MDP_params()
        T = transition_func(
            T=T, height=self.height, width=self.width, prob=prob, params=params
        )

        self.mdp = MDP_2D(S, A, T, R, self.gamma)

        return self.mdp
