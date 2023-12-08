import numpy as np
import torch

from .inference import (
    likelihood,
)  # to fix circular import we can not import log_likelihood_torch here TODO make this prettier

# from .inference.likelihood import log_likelihood_torch


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


def soft_q_iteration_torch(
    R: torch.Tensor,  # R is a one-dimensional tensor with shape (n_states,)
    T_agent: torch.Tensor,
    gamma: float,
    beta: float,  # Inverse temperature parameter for the softmax function
    tol: float = 1e-6,
) -> torch.Tensor:
    n_states, n_actions, _ = T_agent.shape
    V = torch.zeros(n_states)
    Q = torch.zeros((n_states, n_actions))
    policy = torch.zeros((n_states, n_actions))

    while True:
        for s in range(n_states):
            for a in range(n_actions):
                # Calculate the Q-value for action a in state s
                Q[s, a] = R[s] + gamma * torch.dot(T_agent[s, a], V)

        # Apply softmax to get a probabilistic policy
        max_Q = torch.max(Q, axis=1, keepdim=True)[0]
        exp_Q = torch.exp(beta * (Q - max_Q))  # Subtract max_Q for numerical stability
        policy = exp_Q / torch.sum(exp_Q, axis=1, keepdim=True)

        # Calculate the value function V using the probabilistic policy
        V_new = torch.sum(policy * Q, axis=1)

        # Check for convergence
        if torch.max(torch.abs(V - V_new)) < tol:
            break

        V = V_new

    return policy


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
                likelihood.log_likelihood_torch(T_true, policy, traj)
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
