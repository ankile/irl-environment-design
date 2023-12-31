from collections import deque

import numpy as np
import torch


# @jit(nopython=True)
def make_absorbing(absorbing_states, T: np.ndarray) -> None:
    # Now all states with non-zero rewards are absorbing states
    T[absorbing_states, :, :] = 0
    T[absorbing_states, :, absorbing_states] = 1


def transition_matrix(N, M, p, absorbing_states):
    n_states = N * M
    n_actions = 4  # N, E, S, W

    # Initialize the transition matrix T(s, a, s')
    T = np.zeros((n_states, n_actions, n_states))

    # Helper function to convert 2D grid indices to 1D state index
    def to_s(i, j):
        return i * M + j

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
    make_absorbing(absorbing_states, T)

    return T


def insert_walls_into_T(T, wall_indices):
    """
    Insert walls at predefined states into a transition matrix T.

    :param T: The transition matrix with shape (n_states, n_actions, n_states).
    :param wall_indices: indices of the states where the walls should be inserted. Assumes that the indices are flattened.
    :return: The modified transition matrix with walls inserted.
    """

    n_states, n_actions, _ = T.shape
    T = T.copy()

    # Set the transition probabilities into the wall states to zero.
    for s in wall_indices:
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

    return T


def insert_random_walls_into_transition_matrix(
    T, n_walls, absorbing_states, start_state=0
):
    """
    Randomly inserts wall blocks into the transition matrix T.

    :param T: The transition matrix with shape (n_states, n_actions, n_states).
    :param n_walls: The number of walls (cells with zero transition probability) to insert.
    :return: The modified transition matrix with walls inserted.
    """
    n_states, _, _ = T.shape
    T = T.copy()

    # Enumerate all states that can have a wall (i.e. states with reward = 0 and not the start state)
    # wall_candidates = np.where((absorbing_states) & (np.arange(n_states) != start_state))[0] #paul: changed R <= 0 to R == 0 to not insert
    # rewards in states with negative reward
    wall_candidates = np.delete(np.arange(n_states), absorbing_states)
    wall_candidates = np.delete(wall_candidates, start_state)

    # Ensure we're not inserting more walls than there are states.
    n_walls = min(n_walls, len(wall_candidates))

    # Randomly select states to turn into walls.
    wall_states = np.random.choice(wall_candidates, size=n_walls, replace=False)

    # insert walls into transition matrix
    T = insert_walls_into_T(T=T, wall_indices=wall_states)

    return T, wall_states


class Environment:
    def __init__(
        self,
        N,
        M,
        T_true,
        wall_states,
        R_sample_mean,
        start_state,
        n_walls,
        R_true=None,
    ):
        self.N = N
        self.M = M
        self.T_true = T_true
        self.wall_states = wall_states
        self.R_sample_mean = R_sample_mean
        self.n_walls = n_walls
        self.start_state = start_state

        self.R_true = R_true
        self.trajectories = None
        self.regret = None
        self.log_regret = None
        self.max_likelihood_policy = None
        self.id = None


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


def get_candidate_environments(n_envs, N, M, T_true, goal_states) -> list[Environment]:
    envs = []
    # goal_states = np.where(R > 0)[0]
    possible_start_states = [0]
    seen_walls = set()

    # pbar = tqdm(desc="Generating candidate environments", total=n_envs)
    n_states = N * M
    while len(envs) < n_envs:
        # Sample random number of walls to insert
        n_walls = np.random.randint(0, n_states // 2)
        start_state = np.random.choice(possible_start_states)

        T_candidate, wall_states = insert_random_walls_into_transition_matrix(
            T_true,
            n_walls=n_walls,
            absorbing_states=goal_states,
            start_state=start_state,
        )

        # Check if we've already seen this wall configuration
        if tuple(sorted(wall_states)) in seen_walls:
            continue

        seen_walls.add(tuple(sorted(wall_states)))
        # Check if the terminal state is reachable
        # remove as we dont know the goal states
        if not is_terminal_reachable(T_candidate, goal_states, start_state=start_state):
            continue
        R_true = (
            None  # we dont know this, we only add this later for visualization purposes
        )
        envs.append(
            Environment(N, M, T_candidate, wall_states, R_true, start_state, n_walls)
        )
        # pbar.update(1)

    return envs
