from multiprocessing import Pool
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from ipdb import set_trace as bp
from numba.typed import List
from pathlib import Path
import random
from matplotlib.patches import Rectangle
from collections import deque

from datetime import datetime
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor


from pathlib import Path
import torch
from tqdm import trange

np.set_printoptions(linewidth=160, precision=2)

ParamTuple = namedtuple("ParamTuple", ["p", "gamma"])
StateTransition = namedtuple("StateTransition", ["s", "a", "s_next"])
p_limits = (0.5, 0.999)
gamma_limits = (0.5, 0.999)


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


@njit
def max_along_axis_1(matrix):
    max_values = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        max_value = matrix[i, 0]
        for j in range(1, matrix.shape[1]):
            if matrix[i, j] > max_value:
                max_value = matrix[i, j]
        max_values[i] = max_value
    return max_values.reshape((matrix.shape[0], 1))


@njit
def sum_along_axis_1_keepdim(matrix):
    sum_values = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        sum_value = 0
        for j in range(matrix.shape[1]):
            sum_value += matrix[i, j]
        sum_values[i] = sum_value
    return sum_values.reshape((matrix.shape[0], 1))


@njit
def sum_along_axis_1(matrix):
    sum_values = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        sum_value = 0
        for j in range(matrix.shape[1]):
            sum_value += matrix[i, j]
        sum_values[i] = sum_value
    return sum_values


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
        # max_Q = max_along_axis_1(Q)
        exp_Q = np.exp(beta * (Q - max_Q))  # Subtract max_Q for numerical stability
        policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        # policy = exp_Q / sum_along_axis_1_keepdim(exp_Q)

        # Calculate the value function V using the probabilistic policy
        V_new = np.sum(policy * Q, axis=1)
        # V_new = sum_along_axis_1(policy * Q)

        # Check for convergence
        if np.max(np.abs(V - V_new)) < tol:
            break

        V = V_new

    return policy


def generate_trajectory(T_true, policy, absorbing_states, start_state=0, max_steps=100):
    trajectory = []
    current_state = start_state
    n_states, n_actions = policy.shape

    while len(trajectory) < max_steps:
        if current_state in absorbing_states:
            trajectory.append((current_state, -1, -1))  # Append the absorbing state
            break
        # Sample an action based on the policy probabilities for the current state
        action_probabilities = policy[current_state]
        chosen_action = np.random.choice(n_actions, p=action_probabilities)

        # Manually sample next_state based on T_true
        next_state = np.random.choice(n_states, p=T_true[current_state, chosen_action])

        trajectory.append((current_state, chosen_action, next_state))
        current_state = next_state

    return np.array(trajectory)


def generate_n_trajectories(
    T_true, policy, absorbing_states, start_state=0, n_trajectories=100, max_steps=100
):
    trajectories = list()
    for _ in range(n_trajectories):
        trajectories.append(
            generate_trajectory(
                T_true,
                policy,
                absorbing_states,
                start_state=start_state,
                max_steps=max_steps,
            )
        )
    return trajectories


def make_absorbing(R: np.ndarray, T: np.ndarray) -> None:
    # Now all states with non-zero rewards are absorbing states
    reward_indices = np.where(R != 0)[0]
    T[reward_indices, :, :] = 0
    T[reward_indices, :, reward_indices] = 1


def transition_matrix(N, M, p, R=None):
    n_states = N * M
    n_actions = 4  # N, E, S, W

    # Initialize the transition matrix T(s, a, s')
    T = np.zeros((n_states, n_actions, n_states))

    # Helper function to convert 2D grid indices to 1D state index
    to_s = lambda i, j: i * M + j

    # Populate the transition matrix
    for i in range(N):
        for j in range(M):
            s = to_s(i, j)

            # Neighboring states
            neighbors = {
                "N": to_s(i - 1, j) if i > 0 else s,
                "E": to_s(i, j + 1) if j < M - 1 else s,
                "S": to_s(i + 1, j) if i < N - 1 else s,
                "W": to_s(i, j - 1) if j > 0 else s,
            }

            # Set transition probabilities
            for a, action in enumerate(["N", "E", "S", "W"]):
                T[s, a, neighbors[action]] = p
                for other_action in set(["N", "E", "S", "W"]) - {action}:
                    T[s, a, neighbors[other_action]] += (1 - p) / 3

    # Make the transition matrix absorbing
    if R is not None:
        make_absorbing(R, T)

    return T


def plot_trajectories(N, M, trajectories, reward_matrix, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(M, N))

    # Plot rewards as heatmap
    im = ax.imshow(
        reward_matrix.reshape(N, M), cmap="viridis", origin="upper", vmin=-10
    )

    # Plot trajectories
    for traj in trajectories:
        x_coords, y_coords = [], []
        for s, _, _ in traj:
            x, y = divmod(s, M)
            x_coords.append(x)
            y_coords.append(y)

        # Plot the trajectories with a color that stands out
        ax.plot(y_coords, x_coords, marker="o", color="white", alpha=0.5)

    # Assuming 'ax' is the Axes object and 'im' is the image or collection you want the colorbar for:
    cbar = plt.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Reward")

    # Restrict the colorbar values

    # To move the x-axis ticks to the top using the Axes object:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")  # To also move the x-axis label if you have one


def plot_value_and_policy(
    value_function, policy, grid_shape, absorbing_states, title="", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=grid_shape[::-1])

    ax.imshow(
        value_function.reshape(grid_shape),
        cmap="viridis",
        origin="upper",
        interpolation="nearest",
    )
    # Make colorbar
    cbar = ax.figure.colorbar(ax.get_images()[0])

    for s, val in enumerate(value_function):
        if s in absorbing_states:
            continue  # Skip arrows for absorbing states

        row, col = divmod(s, grid_shape[1])
        action = policy[s]
        if action == 0:  # N
            dx, dy = 0, -0.4
        elif action == 1:  # E
            dx, dy = 0.4, 0
        elif action == 2:  # S
            dx, dy = 0, 0.4
        elif action == 3:  # W
            dx, dy = -0.4, 0
        # make arrow
        ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, fc="r", ec="r")

    ax.set_title(title)


def det_to_prob_pol(policy, smoothing=0.0):
    policy = np.eye(n_actions)[policy]
    policy += smoothing
    policy /= policy.sum(axis=1, keepdims=True)
    return policy


# Make function to calculate log-likelihood of a trajectory given a transition matrix and policy
@njit
def compute_log_likelihood(T, policy, trajectory):
    log_likelihood = 0.0
    for s, a, next_s in trajectory[:-1]:
        log_likelihood += np.log(T[s, a, next_s] * policy[s, a])
    return log_likelihood


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
        exp_Q = torch.exp(beta * (Q - max_Q))  # Subtract max_Q for numerical stability
        policy = exp_Q / torch.sum(exp_Q, axis=1, keepdims=True)

        mean_log_likelihood = torch.stack(
            [log_likelihood_torch(T_true, policy, traj) for traj in trajectories]
        ).mean()
        (-mean_log_likelihood).backward()
        optimizer.step()

        # Check for convergence
        if torch.max(torch.abs(policy - old_pi)) < 1e-3:
            break

        old_pi = policy.detach()

    policy = torch.softmax(Q.detach(), dim=1)

    return policy.numpy()


def run_likelihood_exp(small_reward=1, big_reward=4, N=3, M=3, seed=69):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    n_states, n_actions = N * M, 4
    R = np.zeros((N, M))

    R[-1, 0] = small_reward
    R[-1, -1] = big_reward

    R = R.flatten()

    T_true = transition_matrix(N, M, p=0.9)  # True transition matrix
    T_agent_ground = transition_matrix(N, M, p=0.9)
    absorbing_states = np.where(R > 0)[0]  # Absorbing states
    make_absorbing(R, T_true)
    make_absorbing(R, T_agent_ground)

    posteriors = np.linspace(0.5, 0.9, 2)
    # policies = []
    all_trajectories = []
    star_likelihoods = []

    # Make trajectory for posteriors
    for posterior in posteriors:
        policy = soft_q_iteration(R, T_agent_ground, gamma=posterior, beta=10.0)
        # _, policy = value_iteration_with_policy(R, T_agent_ground, gamma=posterior)
        # policy = det_to_prob_pol(policy, smoothing=0.0)

        # Generate trajectories
        trajectories = generate_n_trajectories(
            T_true, policy, absorbing_states, n_trajectories=5, max_steps=100
        )

        all_trajectories += trajectories
        # Calculate likelihood of trajectories
        star_likelihoods += [
            compute_log_likelihood(T_true, policy, traj) for traj in trajectories
        ]

    plot_trajectories(N, M, all_trajectories, R)

    max_likelihood_policy = grad_policy_maximization(
        n_states,
        n_actions,
        all_trajectories,
        T_true,
    )

    # Now use the max_likelihood_policy to calculate the likelihood of each of the trajectories so we can normalize with the star likelihoods
    likelihoods = [
        compute_log_likelihood(T_true, max_likelihood_policy, traj)
        for traj in all_trajectories
    ]

    likelihood_array = np.array([star_likelihoods, likelihoods]).T

    return likelihood_array


def insert_walls_into_transition_matrix(T, n_walls, start_state=0):
    """
    Randomly inserts wall blocks into the transition matrix T.

    :param T: The transition matrix with shape (n_states, n_actions, n_states).
    :param n_walls: The number of walls (cells with zero transition probability) to insert.
    :return: The modified transition matrix with walls inserted.
    """
    n_states, n_actions, _ = T.shape
    T = T.copy()

    # Enumerate all states that can have a wall (i.e. states with reward = 0 and not the start state)
    wall_candidates = np.where((R <= 0) & (np.arange(n_states) != start_state))[0]

    # Ensure we're not inserting more walls than there are states.
    n_walls = min(n_walls, len(wall_candidates))

    # Randomly select states to turn into walls.
    wall_states = np.random.choice(wall_candidates, size=n_walls, replace=False)

    # Set the transition probabilities into the wall states to zero.
    for s in wall_states:
        for a in range(n_actions):
            # Zero out all transitions leading into the wall state.
            T[:, a, s] = 0

            # Zero out all transitions leading out of the wall state.
            T[s, a, :] = 0

    # After modifying the transition probabilities, we need to re-normalize the transition
    # probabilities for each state and action to ensure they still sum to 1.
    for s in range(n_states):
        for a in range(n_actions):
            prob_sum = T[s, a].sum()
            if prob_sum > 0:
                T[s, a] /= prob_sum

    return T, wall_states


def plot_environment(reward_function, wall_states, start_state=(0, 0), ax=None):
    # Assume the reward function is already reshaped to a 2D grid
    N, M = reward_function.shape
    # Identify wall states is the indixes into the

    wall_states = set([(s // M, s % M) for s in wall_states])

    if ax is None:
        fig, ax = plt.subplots()

    ax.matshow(reward_function, cmap=plt.cm.Wistia)

    # Annotate each cell with the reward, start, and wall
    for (i, j), val in np.ndenumerate(reward_function):
        if (i, j) == start_state:
            ax.text(j, i, "Start", va="center", ha="center")
        elif (i, j) in wall_states:
            # Add a dark gray rectangle to represent the wall
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="darkgray"))
        else:
            ax.text(j, i, f"{val:.2f}", va="center", ha="center")


# This function is a standin for the BIRL procedure that will create a proper posterior sampling
def get_parameter_sample(n_samples):
    n_sqrt = int(np.sqrt(n_samples))
    ps = np.linspace(0.5, 0.999, n_sqrt)
    gammas = np.linspace(0.5, 0.999, n_sqrt)

    return list(product(ps, gammas))


class Environment:
    def __init__(self, N, M, R, T_true, wall_states, start_state=0):
        self.N = N
        self.M = M
        self.R = R
        self.T_true = T_true
        self.wall_states = wall_states
        self.start_state = start_state

        self.trajectories = None
        self.regret = None
        self.log_regret = None
        self.max_likelihood_policy = None


def is_terminal_reachable(T, goal_states, start_state=0):
    """
    Check if any of the terminal states are reachable from the top-left state (0, 0)
    using breadth-first search (BFS).

    :param T: The transition matrix with shape (n_states, n_actions, n_states)
    :param terminal_states: A list of terminal states
    :return: True if any terminal state is reachable, False otherwise
    """
    n_states, n_actions, _ = T.shape
    visited = [False] * n_states  # Keep track of visited states
    queue = deque([start_state])  # Start BFS from the top-left state (index 0)
    visited[0] = True

    while queue:
        current_state = queue.popleft()

        # If the current state is a terminal state, return True
        if current_state in goal_states:
            return True

        # Add all reachable states from the current state to the queue
        for a in range(n_actions):
            for s_next in range(n_states):
                if T[current_state, a, s_next] > 0 and not visited[s_next]:
                    visited[s_next] = True
                    queue.append(s_next)

    # If BFS completes without finding a terminal state, return False
    return False


def get_candidate_environments(
    n_envs, N, M, T_true, R, randomize_start_state=True
) -> list[Environment]:
    envs = []
    goal_states = np.where(R > 0)[0]
    possible_start_states = np.where(R == 0)[0] if randomize_start_state else [0]
    seen_walls = set()

    pbar = tqdm(desc="Generating candidate environments", total=n_envs)

    while len(envs) < n_envs:
        # Sample random number of walls to insert
        n_walls = np.random.randint(0, n_states // 2)
        start_state = np.random.choice(possible_start_states)

        T_candidate, wall_states = insert_walls_into_transition_matrix(
            T_true, n_walls=n_walls, start_state=start_state
        )

        # Check if we've already seen this wall configuration
        if tuple(sorted(wall_states)) in seen_walls:
            continue

        seen_walls.add(tuple(sorted(wall_states)))

        # Check if the terminal state is reachable
        if not is_terminal_reachable(T_candidate, goal_states, start_state=start_state):
            continue

        envs.append(Environment(N, M, R, T_candidate, wall_states, start_state))
        pbar.update(1)

    return envs


def environment_search(
    N,
    M,
    R,
    T_true,
    n_env_samples,
    posterior_sample,
    n_traj_per_sample,
) -> list[Environment]:
    # Create the true transition matrix
    absorbing_states = np.where(R != 0)[0]  # Absorbing states

    # 2. Sample $n$ parameter tuples from the prior
    # Now done outside of this function

    # 3. Generate $m$ different candidate environments
    candidate_envs = get_candidate_environments(
        n_env_samples, N, M, T_true, R, randomize_start_state=False
    )

    args = [
        (env, posterior_sample, n_traj_per_sample, N, M, R, T_true, absorbing_states)
        for env in candidate_envs
    ]

    # Parallel processing with multiprocessing and tqdm
    with Pool(64) as pool:
        candidate_envs = list(
            tqdm(pool.imap(process_candidate_env, args), total=len(args), leave=False)
        )

    # 5. Return the environments (ordered by regret, with higest regret first)
    return sorted(candidate_envs, key=lambda env: env.regret, reverse=True)


def process_candidate_env(args):
    (
        candidate_env,
        posterior_sample,
        n_traj_per_sample,
        N,
        M,
        R,
        T_true,
        absorbing_states,
    ) = args
    policies = []
    trajectories = []
    likelihoods = []
    n_states, n_actions = N * M, 4

    for p, gamma in posterior_sample:
        # 4.1.1 Find the optimal policy for this env and posterior sample
        T_agent = transition_matrix(N, M, p=p, R=R)
        policy = soft_q_iteration(R, T_agent, gamma=gamma, beta=100.0)
        policies.append(policy)

        # 4.1.2 Generate $m$ trajectories from this policy
        policy_traj = generate_n_trajectories(
            candidate_env.T_true,
            policy,
            absorbing_states,
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
        n_states,
        n_actions,
        trajectories,
        T_true,
        n_iter=100,
    )
    candidate_env.max_likelihood_policy = most_likely_policy

    # 4.3 Calculate the regret of the most likely policy
    most_likely_likelihoods = [
        compute_log_likelihood(T_true, most_likely_policy, traj)
        for traj in trajectories
    ]

    all_likelihoods = np.array([likelihoods, most_likely_likelihoods]).T
    candidate_env.log_likelihoods = all_likelihoods.mean(axis=0)
    candidate_env.log_regret = np.diff(candidate_env.log_likelihoods).item()

    all_likelihoods = np.exp(all_likelihoods)
    candidate_env.likelihoods = all_likelihoods.mean(axis=0)
    candidate_env.regret = -np.diff(candidate_env.likelihoods).item()

    candidate_env.trajectories = trajectories

    return candidate_env


def plot_environments_with_regret(envs):
    N, M, R = envs[0].N, envs[0].M, envs[0].R

    # Plot all the envs with their regrets
    for env in sorted(envs, key=lambda x: x.regret, reverse=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot_environment(
            R.reshape(N, M),
            env.wall_states,
            start_state=(env.start_state // M, env.start_state % M),
            ax=ax1,
        )

        # Show the trajectories
        plot_trajectories(N, M, env.trajectories, R, ax=ax2)

        # Remove the colorbar from the second plot
        ax2.get_images()[0].colorbar.remove()

        fig.suptitle(
            f"Regret: {env.regret:.3f}, Log regret: {env.log_regret:.3f}\n({env.likelihoods} / {env.log_likelihoods})"
        )


def prior_sample() -> ParamTuple:
    p = np.random.uniform(*p_limits)
    gamma = np.random.uniform(*gamma_limits)

    return ParamTuple(p, gamma)


def parameter_proposal(previous_sample: ParamTuple, step_size: float) -> ParamTuple:
    p = np.random.normal(previous_sample.p, step_size)
    p = np.clip(p, *p_limits)

    gamma = np.random.normal(previous_sample.gamma, step_size)
    gamma = np.clip(gamma, *gamma_limits)

    return ParamTuple(p, gamma)


from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def parameter_proposal_truncnorm(
    previous_sample: ParamTuple,
    step_size: float,
) -> ParamTuple:
    # Truncated normal distribution for p
    p_dist = get_truncated_normal(
        mean=previous_sample.p, sd=step_size, low=p_limits[0], upp=p_limits[1]
    )
    p = p_dist.rvs()

    # Truncated normal distribution for gamma
    gamma_dist = get_truncated_normal(
        mean=previous_sample.gamma,
        sd=step_size,
        low=gamma_limits[0],
        upp=gamma_limits[1],
    )
    gamma = gamma_dist.rvs()

    return ParamTuple(p, gamma)


def expert_trajectory_likelihood(
    parameter_sample: ParamTuple,
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
) -> float:
    log_likelihood = 0.0

    for env, trajectories in expert_trajectories:
        T_agent = transition_matrix(env.N, env.M, p=parameter_sample.p, R=env.R)
        policy = soft_q_iteration(
            env.R, T_agent, gamma=parameter_sample.gamma, beta=20.0
        )
        for traj in trajectories:
            log_likelihood += compute_log_likelihood(env.T_true, policy, traj)

    return np.exp(log_likelihood)


def bayesian_parameter_learning(
    # TODO: Find an appropriate data structure for expert trajectories
    # It needs to account for the possibility of multiple trajectories per environment
    expert_trajectories: list[tuple[Environment, list[StateTransition]]],
    sample_size: int,
    previous_sample: ParamTuple = None,
):
    # Samples from the posterior
    posterior_samples: list[ParamTuple] = []
    n_accepted = 0
    step_size = 0.1

    # Start the chain at the previous sample if provided, otherwise sample from the prior
    if previous_sample is None:
        previous_sample = prior_sample()

    old_likelihood = expert_trajectory_likelihood(previous_sample, expert_trajectories)

    it = trange(sample_size, desc="Posterior sampling", leave=False)
    for k in it:
        # Create a new proposal for (p_i, gamma_i)
        # proposed_parameter: ParamTuple = parameter_proposal(
        #     previous_sample, step_size=step_size
        # )
        proposed_parameter: ParamTuple = parameter_proposal_truncnorm(
            previous_sample, step_size=step_size
        )
        likelihood = expert_trajectory_likelihood(
            proposed_parameter, expert_trajectories
        )

        # Check if we accept the proposal
        p = likelihood  # We don't multiply by the prior because it's uniform
        p_old = old_likelihood
        quotient = p / p_old
        if np.random.uniform(0, 1) < quotient:
            previous_sample = proposed_parameter
            old_likelihood = likelihood
            n_accepted += 1
        posterior_samples.append(previous_sample)

        # Based on current acceptance rates, adjust step size and n_steps
        acceptance_rate = n_accepted / (k + 1)
        if acceptance_rate > 0.25:
            step_size = round(min(1, step_size + 0.01), 3)
        elif acceptance_rate < 0.21:
            step_size = round(max(0.01, step_size - 0.01), 3)

        it.set_postfix(
            {
                "Acceptance rate": round(100 * acceptance_rate, 1),
                "step_size": step_size,
            }
        )

    return posterior_samples


def plot_posterior_distribution(
    posterior_samples: list[ParamTuple], true_params: ParamTuple = None, ax=None
):
    """
    Plot the join distribution of p and gamma from the posterior samples as a 2D histogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    # Unzipping the list of tuples
    p_values, gamma_values = zip(*(posterior_samples))

    # Plotting the 2D distribution
    ax.scatter(p_values, gamma_values, alpha=0.3)
    ax.set_title("Posterior distribution")
    ax.set_xlabel("$p_i$")
    ax.set_ylabel("$\\gamma_i$")
    ax.grid(True)
    ax.set_xlim(p_limits)
    ax.set_ylim(gamma_limits)

    if true_params is not None:
        ax.scatter(
            true_params.p,
            true_params.gamma,
            marker="*",
            color="red",
            label="True parameters",
        )
        ax.legend()


def save_env_regret_plots(round_idx, envs: list[Environment], result_save_path: Path):
    # Save the plots
    folder_path = result_save_path / "regret_plots" / str(round_idx)
    folder_path.mkdir(exist_ok=True, parents=True)

    for i, env in enumerate(envs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot_environment(
            env.R.reshape(env.N, env.M),
            env.wall_states,
            start_state=(env.start_state // env.M, env.start_state % env.M),
            ax=ax1,
        )

        # Show the trajectories
        plot_trajectories(env.N, env.M, env.trajectories, env.R, ax=ax2)

        # Remove the colorbar from the second plot
        ax2.get_images()[0].colorbar.remove()

        fig.suptitle(
            f"Regret: {env.regret:.3f}, Log regret: {env.log_regret:.3f}\n({env.likelihoods} / {env.log_likelihoods})"
        )

        plt.savefig(str(folder_path / f"{env.regret:.3f}-{i}.png"))
        plt.close()


def save_env_traj_posterior_plots(
    round_idx: int,
    env: Environment,
    posterior_samples: list[ParamTuple],
    true_params: ParamTuple,
    result_save_path: Path,
):
    # Save the plots
    folder_path = result_save_path / "traj_posterior_plots"
    folder_path.mkdir(exist_ok=True, parents=True)

    # Three plots: 1. The chosen env; 2. The rolled out trajectories in it; 3. The posterior distribution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    plot_environment(
        env.R.reshape(env.N, env.M),
        env.wall_states,
        start_state=(env.start_state // env.M, env.start_state % env.M),
        ax=ax1,
    )

    # Show the trajectories
    plot_trajectories(env.N, env.M, env.trajectories, env.R, ax=ax2)

    # Remove the colorbar from the second plot
    ax2.get_images()[0].colorbar.remove()

    # Plot the posterior on ax3
    plot_posterior_distribution(posterior_samples, true_params=true_params, ax=ax3)

    fig.suptitle(f"Round {round_idx}")

    plt.savefig(str(folder_path / f"round-{round_idx}.png"))
    plt.close()


def environment_design_experiment(
    n_rounds: int,
    traj_per_round: int,
    n_env_samples: int,
    n_posterior_samples: int,
    base_env: Environment,
    true_params: ParamTuple,
    result_save_path=None,
):
    if result_save_path is not None:
        result_save_path = result_save_path / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        result_save_path.mkdir(exist_ok=True, parents=True)

    expert_trajectories = []
    previous_sample = None

    # Start with an initial sample from the prior
    posterior_samples = bayesian_parameter_learning(
        expert_trajectories, n_posterior_samples, previous_sample=previous_sample
    )
    # Remove burn-in
    posterior_samples = posterior_samples[-1_000:]

    for i in trange(n_rounds, desc="Rounds"):
        # Sample a subset of the posterior samples
        sample_idxs = np.random.choice(
            np.arange(len(posterior_samples)), size=100, replace=False
        )
        samples = [posterior_samples[i] for i in sample_idxs]

        # Find the env with the highest regret to observe the expert in
        envs = environment_search(
            base_env.N,
            base_env.M,
            base_env.R,
            base_env.T_true,
            n_env_samples,
            samples,
            traj_per_round,
        )

        # Generate `traj_per_round` trajectories in the env with the highest regret
        # env: Environment = sorted(envs, key=lambda x: x.log_regret, reverse=True)[0]
        env: Environment = sorted(envs, key=lambda x: x.regret, reverse=True)[0]

        T_agent = transition_matrix(env.N, env.M, p=true_params.p, R=env.R)
        agent_policy = soft_q_iteration(
            env.R, T_agent, gamma=true_params.gamma, beta=20.0
        )

        # Generate trajectories
        absorbing_states = np.where(R != 0)[0]  # Absorbing states
        trajectories = generate_n_trajectories(
            env.T_true,
            agent_policy,
            absorbing_states,
            start_state=env.start_state,
            n_trajectories=traj_per_round,
            max_steps=(env.N + env.M - 2) * 2,
        )

        expert_trajectories.append((env, trajectories))

        previous_sample = posterior_samples[-1]

        # Start with an initial sample from the prior
        posterior_samples = bayesian_parameter_learning(
            expert_trajectories, n_posterior_samples, previous_sample=previous_sample
        )
        # Remove burn-in
        posterior_samples = posterior_samples[-1_000:]

        # If we have a save path, save a figure with the environment, trajectories, and posterior
        if result_save_path is not None:
            save_env_regret_plots(i, envs, result_save_path)
            save_env_traj_posterior_plots(
                i, env, posterior_samples, true_params, result_save_path
            )

    return expert_trajectories, posterior_samples


if __name__ == "__main__":
    agent_p = 0.7
    agent_gamma = 0.7
    true_params = ParamTuple(agent_p, agent_gamma)

    # Run the experiment
    n_env_samples = 100
    n_posterior_samples = 2_000
    n_traj_per_sample = 20

    ## 0.2 Setup the environment
    N, M = 6, 6
    n_states, n_actions = N * M, 4

    # Create a type of BigSmall world with a dangerous zone
    R = np.zeros((N, M))
    R[-1, 0] = 5
    R[-1, 1] = 10
    R[-1, 2] = 15
    R[-1, 3] = 20
    R[-1, 4] = 25
    R[-1, 5] = 30

    R[0, -1] = 10
    R[2, -1] = 20

    R[1, -2] = -1
    R[1, -3] = -1
    R[3, -2] = -1
    R[3, -3] = -1
    R = R.flatten()

    p_true = 0.95
    T_true = transition_matrix(N, M, p=p_true, R=R)

    base_env = Environment(N, M, R, T_true, wall_states=[])

    result_save_path = Path("results")

    expert_trajectories, posterior_samples = environment_design_experiment(
        n_rounds=3,
        traj_per_round=2,
        n_env_samples=n_env_samples,
        n_posterior_samples=n_posterior_samples,
        base_env=base_env,
        true_params=true_params,
        result_save_path=result_save_path,
    )
