import numpy as np
import torch
from numba import jit



# @njit
# @jit(nopython=True)
def soft_q_iteration(
    R: np.ndarray,
    T: np.ndarray,
    gamma: float,
    beta: float,
    tol: float = 1e-6,
    return_what = "policy",
    Q_init = None,
    V_init = None,
    policy_init = None,
    verbose = False
) -> np.ndarray:
    
    '''
    Computes a policy, V-Function or Q-Function using a Boltzmann-like policy.

    Args:
    - R (np.ndarray): The reward function R of shape (n_states,)
    - T (np.ndarray): The transition matrix T of shape (n_states, n_actions, n_states)
    - gamma (float): The discount factor gamma
    - beta (float): Inverse temperature parameter for the Boltzmann policy
    - tol (float): Tolerance for convergence
    - return_what (str): What to return. Choose from "policy", "Q", "V"
    - Q_init (np.ndarray): Initialization of the Q-Function of shape (n_states, n_actions), optional.
    - V_init (np.ndarray): Initialization of the V-Function of shape (n_states,), optional.
    - policy_init (np.ndarray): Initialization of the policy of shape (n_states, n_actions), optional.

    Returns one of:
    - policy (np.ndarray): The policy of shape (n_states, n_actions)
    - Q (np.ndarray): The Q-Function of shape (n_states, n_actions)
    - V (np.ndarray): The V-Function of shape (n_states,)
    '''

    n_states, n_actions, _ = T.shape


    if Q_init is not None:
        Q = Q_init
    else:
        Q = np.zeros((n_states, n_actions))

    if V_init is not None:
        V = V_init
    else:
        V = np.zeros(n_states)

    if policy_init is not None:
        policy = policy_init
    else:
        policy = np.zeros((n_states, n_actions))

    _n_iter = 0

    while True:

        # if verbose:
        #     print("V: ", V)
        #     print("R: ", R)
        #     print("T: ", T)
        #     print("gamma: ", gamma)

        Q = np.einsum("ijk, k-> ij", T, R+gamma*V)

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
            print("Error: ", np.max(np.abs(V - V_new)))
            print("V: ", V)
            print("R: ", R)
            print("T: ", T)
            print("gamma: ", gamma)



    if return_what =="all":
        return policy, Q, V
    elif return_what == "policy":
        return policy
    elif return_what == "Q":
        return Q
    elif return_what =="V":
        return V
    else:
        raise ValueError("Invalid return_what argument. Choose 'all', 'policy', 'Q', or 'V'. You gave: ", return_what)



def soft_bellman_update_V(R, gamma, T, V):
    
    if torch.isnan(V).any():
        print("V contains NaN values.")
        print("V: ", V)
        print("R: ", R)
        print("T: ", T)

    #TODO get Nan values in wall states. Overwrite by zero. Fix this.
    return torch.log(torch.sum(torch.exp(torch.matmul(T, R+gamma*V)), axis=1))
    # _Q_Values = torch.einsum("ijk, k-> ij", T, R+gamma*V)
    # _Q_Values = _Q_Values - torch.max(_Q_Values, axis=1, keepdims=True)[0] #Subtract max_Q for numerical stability
    # return torch.log(torch.einsum("ij-> i", torch.exp(_Q_Values)))



def soft_bellman_FP_V(R, gamma, T, V):
    #TODO get Nan values in wall states. Overwrite by zero. Fix this.
    return soft_bellman_update_V(R, gamma, T, V) - V


def soft_V_iteration_torch(R, gamma, T, V_init=None, tol=1e-4):
    if V_init is None:
        V_init = torch.zeros_like(R)

    V = V_init

    n_iterations = 0
    
    while True:

        n_iterations += 1
        V_new = soft_bellman_update_V(R, gamma,T, V)
        if torch.max(torch.abs(V - V_new)) < tol:
            break

        if n_iterations % 1_000 == 0:
            print("Warning: Soft V-iteration did not converge within 1_000 steps.")
            print("Error: ", torch.max(torch.abs(V - V_new)))

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
    V_star = soft_V_iteration_torch(R, gamma, T, V_init=V, tol=1e-3)

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


'''
Legacy code for likelihood maximization in the original Environment Design approach when we replace the value function by the
likelihood function.
'''

def log_likelihood_torch(T, policy, trajectory):
    log_likelihood = torch.tensor(0.0)
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += torch.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


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