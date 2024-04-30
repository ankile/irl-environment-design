import numpy as np
import torch
from numba import jit


def log_likelihood_torch(T, policy, trajectory):
    log_likelihood = torch.tensor(0.0)
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += torch.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


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
# @jit(nopython=True)
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
    _n_iter = 0

    while True:
        Q = np.einsum("ijk, k-> ij", T_agent, R+gamma*V)

        # Apply softmax to get a probabilistic policy
        max_Q = np.max(Q, axis=1, keepdims=True)

        # Subtract max_Q for numerical stability
        exp_Q = np.exp(beta * (Q - max_Q))
        policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        # Calculate the value function V using the probabilistic policy
        V_new = np.sum(policy * Q, axis=1)

        # Check for convergence
        if np.max(np.abs(V - V_new)) < tol:
            break

        V = V_new
        _n_iter += 1

        if _n_iter == 1_000:
            print("Warning: Soft Q-iteration did not converge within 1_000 steps.")

    return policy


# def soft_q_iteration_torch(
#     R: torch.Tensor,  # R is a one-dimensional tensor with shape (n_states,)
#     T_agent: torch.Tensor,
#     gamma: float,
#     beta: float,  # Inverse temperature parameter for the softmax function
#     tol: float = 1e-6,
# ) -> torch.Tensor:
#     n_states, n_actions, _ = T_agent.shape
#     V = torch.zeros(n_states)
#     Q = torch.zeros((n_states, n_actions))
#     policy = torch.zeros((n_states, n_actions))

#     while True:
#         for s in range(n_states):
#             for a in range(n_actions):
#                 # Calculate the Q-value for action a in state s
#                 Q[s, a] = R[s] + gamma * torch.dot(T_agent[s, a], V)

#         # Apply softmax to get a probabilistic policy
#         max_Q = torch.max(Q, axis=1, keepdim=True)[0]
#         exp_Q = torch.exp(beta * (Q - max_Q))  # Subtract max_Q for numerical stability
#         policy = exp_Q / torch.sum(exp_Q, axis=1, keepdim=True)

#         # Calculate the value function V using the probabilistic policy
#         V_new = torch.sum(policy * Q, axis=1)

#         # Check for convergence
#         if torch.max(torch.abs(V - V_new)) < tol:
#             break

#         V = V_new

#     return policy


def grad_policy_maximization(
    n_states, n_actions, trajectories, T_true, beta=10, n_iter=1_000
):
    Q = torch.zeros(n_states, n_actions, requires_grad=True)

    optimizer = torch.optim.Adam([Q], lr=0.1)
    T_true = torch.tensor(T_true)
    old_pi = torch.zeros(n_states, n_actions)

    for _ in range(n_iter):
        optimizer.zero_grad()

        # Derive the policy from the Q-function
        # Apply softmax to get a probabilistic policy
        max_Q = torch.max(Q, axis=1, keepdims=True)[0]
        # max_Q = max_along_axis_1(Q)
        # Subtract max_Q for numerical stability
        exp_Q = torch.exp(beta * (Q - max_Q))
        policy = exp_Q / torch.sum(exp_Q, axis=1, keepdims=True)

        mean_log_likelihood = torch.stack(
            [
                log_likelihood_torch(T_true, policy, traj)
                for traj in trajectories
            ]
        ).mean()
        (-mean_log_likelihood).backward()
        optimizer.step()

        # Check for convergence
        if torch.max(torch.abs(policy - old_pi)) < 1e-3:
            break

        old_pi = policy.detach()

    policy = torch.softmax(Q.detach(), dim=1)

    return policy.numpy()


def soft_bellman_update_V(R, gamma, T, V):
    return torch.log(torch.sum(torch.exp(torch.matmul(T, R+gamma*V)), axis=1))


def soft_bellman_FP_V(R, gamma, T, V):
    return soft_bellman_update_V(R, gamma, T, V) - V


def soft_V_iteration_torch(R, gamma, T, V_init=None, tol=1e-6):
    if V_init is None:
        V_init = torch.zeros(49)

    V = V_init
    
    while True:
        V_new = soft_bellman_update_V(R, gamma,T, V)
        if torch.max(torch.abs(V - V_new)) < tol:
            break
        V = V_new
    return V


def differentiate_V(R: torch.tensor, gamma: torch.tensor, T: torch.tensor, V: torch.tensor):

    '''
    Calculate the derivative of the value function V with respect to the reward function R and the transition matrix T via implicit differentiation.

    Args:
    - R (torch.tensor): The reward function R of shape (n_states,)
    - gamma (torch.tensor): The discount factor gamma
    - T (torch.tensor): The transition matrix T of shape (n_states, n_actions, n_states)
    - V (torch.tensor): Initialization of the value function V of shape (n_states,), optional.

    Returns:
    - V_star (torch.tensor): The optimal value function V, shape (n_states,)
    - R_grad_out (torch.tensor): The gradient of the value function V with respect to the reward function R, shape (n_states,) evaluated at V_star
    - T_grad_out (torch.tensor): The gradient of the value function V with respect to the transition matrix T, shape (n_states, n_actions, n_states) evaluated at V_star

    Notes:
    - We replaced the hard max in the Bellman operator with a soft max to make the operator differentiable. Thereby, we no longer converge to the "true" value function
    but rather to an approximation.
    '''

    #Perform value iteration to find a fixed point of the Bellman operator.
    V_star = soft_V_iteration_torch(R, gamma, T, V_init=V, tol=1e-6)

    #Calculate the gradient of the value function using the implicit function theorem.
    # A closed form expression for V is given by psi.
    R_grad = torch.autograd.Variable(R, requires_grad=True)
    T_grad = torch.autograd.Variable(T, requires_grad=True)
    V_star_grad = torch.autograd.Variable(V_star, requires_grad=True)

    df_dtheta = soft_bellman_FP_V(R = R_grad, gamma=gamma, T=T_grad, V = V_star_grad)
    df_dtheta.backward(torch.ones_like(df_dtheta))

    Jacobian = torch.autograd.functional.jacobian(func=soft_bellman_FP_V, inputs=(R_grad, gamma, T_grad, V_star_grad))
    df_dw = Jacobian[3]
    del Jacobian

    #Inverse of Jacobian.
    df_dw_inv = torch.inverse(df_dw)

    R_grad_out = - torch.einsum('ij,j->i', df_dw_inv, R_grad.grad)
    T_grad_out = - torch.einsum('ij,jkl->ikl', df_dw_inv, T_grad.grad)

    return V_star, R_grad_out, T_grad_out