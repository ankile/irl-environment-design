import random
import copy
import torch

from tqdm import tqdm
from maze_env import ConstructedMazeEnv
from auxiliary.auxiliary import *
from auxiliary.mdp_solver import *
from random import sample  # random order in for-loops of VI

from ipdb import set_trace as bp


def domain_randomisation(env):
    # probability of a wall in a before empty cell
    p = 0.3
    walls = []
    for state in range(env.state_space.n):
        coord = scalar_to_coordinate(env, state)
        cell = env.grid.get(*coord)
        if cell is None and env.agent_start_pos != coord:
            if random.random() < p:
                walls.append(coord)
    return walls


# Maximin Environment Design for maze generation; Computation via Extended Value Iteration
# Input: mean reward function (posterior mean) and samples from the belief (posterior)
def maze_maximin_environment_design(env, m_reward, s_reward, delta=0.0001):
    n_rewards = len(s_reward)
    n_env_actions = 2

    # value functions w.r.t. sampled rewards
    V = np.zeros([n_rewards, env.state_space.n])
    Q = np.zeros([n_rewards, env.state_space.n, env.action_space.n])
    # pol = np.zeros([n_rewards, env.state_space.n, env.action_space.n])

    # value function w.r.t. mean rewards
    V_mean = np.zeros(env.state_space.n)
    Q_mean = np.zeros([env.state_space.n, env.action_space.n])
    # pol_mean = np.zeros([env.state_space.n, env.action_space.n])

    # # burn-in value function
    empty_env = ConstructedMazeEnv(size=env.width, walls=[])
    for i in range(n_rewards):
        empty_env.rewards = s_reward[i]
        V[i], x, y = value_iteration(empty_env)
    empty_env.rewards = m_reward
    V_mean, x, y = value_iteration(empty_env)

    # value function for environment
    V_env = np.zeros(env.state_space.n)
    Q_env = np.zeros([env.state_space.n, n_env_actions])
    pol_env = np.zeros(env.state_space.n)

    walls = []
    current_env = ConstructedMazeEnv(size=env.width, walls=walls)
    iterations = 0
    while True:
        V_old = V.copy()
        V_mean_old = V_mean.copy()
        V_env_old = V_env.copy()
        Q_env = np.zeros([env.state_space.n, n_env_actions])
        iterations += 1
        for next_state in sample(range(env.state_space.n), env.state_space.n):
            # get neighbouring states
            neighbours = np.unique(
                np.array(
                    [
                        next_state - 1,
                        next_state + 1,
                        next_state - (env.width + 1),
                        next_state + (env.width + 1),
                    ]
                ).clip(0, env.state_space.n - 1)
            )
            for env_action in sample(range(n_env_actions), n_env_actions):
                # get current environment
                current_walls = walls.copy()
                next_state_coord = scalar_to_coordinate(env, next_state)
                if env_action == 0 and next_state_coord in current_walls:
                    current_walls.remove(next_state_coord)
                elif (
                    env_action == 1
                    and current_env.grid.get(*next_state_coord) is None
                    and env.agent_start_pos != next_state_coord
                ):
                    current_walls.append(next_state_coord)
                current_env = ConstructedMazeEnv(size=env.width, walls=current_walls)
                for neigh in neighbours:
                    for reward_index in range(n_rewards):
                        temp_Q_rew = np.zeros(env.action_space.n)
                        for action in range(env.action_space.n):
                            temp_Q_rew[action] = env.gamma * np.dot(
                                current_env.get_transition_probabilities(neigh, action),
                                V[reward_index, :],
                            )
                        Q_env[next_state, env_action] += max(temp_Q_rew) / n_rewards
                    # mean reward function
                    temp_Q_rew = np.zeros(env.action_space.n)
                    for action in range(env.action_space.n):
                        temp_Q_rew[action] = env.gamma * np.dot(
                            current_env.get_transition_probabilities(neigh, action),
                            V_mean[:],
                        )
                    Q_env[next_state, env_action] -= max(
                        temp_Q_rew
                    )  # WE CAN ADD A FACTOR < 1 HERE TO PROMOTE SOLVABLE ENVIRONMENTS
            V_env[next_state] = max(Q_env[next_state, :])
            pol_env[next_state] = np.argmax(Q_env[next_state, :])
            walls = []
            for i in np.nonzero(pol_env)[0]:
                walls.append(scalar_to_coordinate(env, i))
            current_env = ConstructedMazeEnv(size=env.width, walls=walls)
            for reward_index in range(n_rewards):
                for s in neighbours:
                    for a in range(env.action_space.n):
                        Q[reward_index, s, a] = s_reward[reward_index][
                            s
                        ] + env.gamma * np.dot(
                            current_env.get_transition_probabilities(s, a),
                            V[reward_index, :],
                        )
                    V[reward_index, s] = max(Q[reward_index, s, :])
            for s in neighbours:
                for a in range(env.action_space.n):
                    Q_mean[s, a] = m_reward[s] + env.gamma * np.dot(
                        current_env.get_transition_probabilities(s, a), V_mean[:]
                    )
                V_mean[s] = max(Q_mean[s, :])
        if (
            np.max(np.abs(V_old - V)) < delta
            and max(np.abs(V_mean_old - V_mean)) < delta
            and max(np.abs(V_env_old - V_env)) < delta
        ):
            print("Max-Min Maze VI Iterations", iterations)
            break
        if iterations % 10 == 0:
            print("Iterations", iterations)
            print("Value wrt Mean", V_mean[env.width + 1])
            print("Value wrt a sampled rew ", V[0, env.width + 1])
            print("Env Value", V_env[env.width + 1])
            it_env = ConstructedMazeEnv(size=env.width, walls=walls)
            save_initial_render(it_env, iterations)
            print(
                "Current Iteration Walls: Regret",
                evaluate_regret_of_maze(env, walls, s_reward, m_reward),
            )
        if iterations > 100:
            break
    return walls


""" Brute-force search for Bayesian regret maximising maze. """


def non_blocking_domain_randomisation(env, prob=0.2):
    # probability of a wall in a before empty cell
    walls = []
    eff_state_space = [
        i
        for i in range(env.state_space.n)
        if (i % env.width != 0)
        and (i % env.width != env.width - 1)
        and (i > env.width)
        and (i < env.width * (env.width - 1))
        and (i != env.width + 2)
        and (i != env.width * 2 + 1)
        and (i != 40)
        and (i != 41)
        and (i != 42)
    ]  # area around starting position is wall-free # 11, 19
    # and (i != env.width + 3) and (i != env.width * 3 + 1)    # 12, 28
    # and (i != env.width * 2 + 2) and (i != env.width * 2 + 3)
    # and (i != env.width * 3 + 2) and (i != env.width * 3 + 3)]
    for state in eff_state_space:
        coord = scalar_to_coordinate(env, state)
        cell = env.grid.get(*coord)
        if cell is None and env.agent_start_pos != coord:
            if random.random() < prob:
                walls.append(coord)
    return walls


# get environment candidates
def get_environment_candidates(env, size):
    candidates = []
    candidates.append([])  # adding empty maze
    candidates.append([[5, 1], [5, 2], [5, 3], [6, 3]])
    candidates.append([[4, 1], [5, 3], [3, 5], [3, 6]])
    candidates.append(
        [[3, 4], [3, 5], [3, 6], [2, 5], [3, 7], [6, 1], [5, 2], [6, 3], [5, 3], [7, 3]]
    )
    size -= len(candidates)

    for _ in range(size // 3):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.1))
    for _ in range(round(size / 3)):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.2))
    for _ in range(math.ceil(size / 3)):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.3))
    return candidates


# Evaluate Bayesian regret of environment w.r.t. sampled rewards and posterior mean
def evaluate_value_regret_of_maze(env, walls, s_reward, m_reward):
    regret = 0
    maze = ConstructedMazeEnv(
        size=env.width, walls=walls
    )  # just change the transition function, maybe
    for reward in s_reward:
        maze.rewards = reward
        V, Q, pol = value_iteration(maze)
        regret += V[env.width + 1] / len(s_reward)
    maze.rewards = m_reward
    V_mean, Q, pol = value_iteration(maze)

    regret -= V_mean[env.width + 1]

    return regret


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
        # Subtract max_Q for numerical stability
        exp_Q = torch.exp(beta * (Q - max_Q))
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


def evaluate_likelihood_regret_of_maze(env, walls, s_reward, m_reward):
    trajectories = []
    likelihoods = []

    maze = ConstructedMazeEnv(size=env.width, walls=walls)

    T_true = maze.P

    for R in s_reward:
        # 4.1.1 Find the optimal policy for this env and posterior sample
        maze.rewards = R
        _, Q, _ = value_iteration(maze)
        Q_exp = np.exp(30 * Q)
        pol = Q_exp / np.sum(Q_exp, axis=1, keepdims=True)

        # 4.1.2 Generate $m$ trajectories from this policy
        policy_traj = [get_expert_trajectory_alt(maze, pol) for _ in range(2)]
        # 4.1.3 Calculate the likelihood of the trajectories
        policy_likelihoods = [
            compute_log_likelihood(T_true, pol, traj) for traj in policy_traj
        ]

        # 4.1.4 Store the trajectories and likelihoods
        trajectories += policy_traj
        likelihoods += policy_likelihoods

    # 4.2 Find the policy with the highest likelihood
    n_states = env.state_space.n
    n_actions = env.action_space.n
    most_likely_policy = grad_policy_maximization(
        n_states=n_states,
        n_actions=n_actions,
        trajectories=trajectories,
        T_true=T_true,
        n_iter=100,
    )

    # 4.3 Calculate the regret of the most likely policy
    most_likely_likelihoods = [
        compute_log_likelihood(T_true, most_likely_policy, traj)
        for traj in trajectories
    ]

    all_likelihoods = np.array([likelihoods, most_likely_likelihoods]).T

    all_likelihoods = np.exp(all_likelihoods)
    likelihoods = all_likelihoods.mean(axis=0)
    regret = -np.diff(likelihoods).item()
    return regret


def evaluate_regret_of_maze(env, walls, s_reward, m_reward):
    return evaluate_value_regret_of_maze(env, walls, s_reward, m_reward)


# brute force
def brute_force_maze_design(env, s_reward, m_reward, candidate_size):
    candidate_walls = get_environment_candidates(env, candidate_size)
    bayes_regret = []
    for idx, walls in enumerate(tqdm(candidate_walls, desc="Brute Force")):
        bayes_regret.append(evaluate_regret_of_maze(env, walls, s_reward, m_reward))
    return candidate_walls, bayes_regret


# brute force
def evaluate_regret_for_candidates(
    env, s_reward, m_reward, candidate_walls, regret_func
):
    bayes_regret = []
    for idx, walls in enumerate(tqdm(candidate_walls, desc="Brute Force")):
        bayes_regret.append(regret_func(env, walls, s_reward, m_reward))
    return bayes_regret


# Brute force, but in parallel 😎
# from multiprocessing import Pool
# from tqdm import tqdm


# def evaluate_single_case(args):
#     env, walls, s_reward, m_reward = args
#     return evaluate_regret_of_maze(env, walls, s_reward, m_reward)


# def brute_force_maze_design(env, s_reward, m_reward, candidate_size, num_processes=4):
#     candidate_walls = get_environment_candidates(env, candidate_size)

#     # Create a pool of workers and map the evaluation function over the candidates
#     with Pool(processes=num_processes) as pool:
#         results = list(
#             tqdm(
#                 pool.imap(
#                     evaluate_single_case,
#                     [(env, w, s_reward, m_reward) for w in candidate_walls],
#                 ),
#                 total=len(candidate_walls),
#             )
#         )

#     bayes_regret = results
#     return candidate_walls, bayes_regret


""" Regret-based environment design via extended value iteration. """


# Regret-Based Environment Design for Maze Environment using P_s
def regret_based_maze_design(env, m_reward, s_reward, delta=0.0001):
    n_rewards = len(s_reward)
    n_env_actions = 4  # action 0 ~ right and below no wall, action 1 ~ right wall, below no wall, action 2 ~ right no wall, below wall, action 3 ~ right and below wall

    # value functions w.r.t. sampled rewards
    V = np.zeros([n_rewards, env.state_space.n])
    Q = np.zeros([n_rewards, env.state_space.n, env.action_space.n])
    # pol = np.zeros([n_rewards, env.state_space.n, env.action_space.n])

    # value function w.r.t. mean rewards
    V_mean = np.zeros(env.state_space.n)
    Q_mean = np.zeros([env.state_space.n, env.action_space.n])
    # pol_mean = np.zeros([env.state_space.n, env.action_space.n])

    # # burn-in value function
    empty_env = ConstructedMazeEnv(size=env.width, walls=[])
    for i in range(n_rewards):
        empty_env.rewards = s_reward[i]
        V[i], x, y = value_iteration(empty_env)
    empty_env.rewards = m_reward
    V_mean, x, y = value_iteration(empty_env)

    # value function for environment
    V_env = np.zeros(env.state_space.n)
    Q_env = np.zeros([env.state_space.n, n_env_actions])
    pol_env = np.zeros(env.state_space.n)

    max_regret = 0
    walls = []
    current_env = ConstructedMazeEnv(size=env.width, walls=walls)
    iterations = 0
    while True:
        V_old = V.copy()
        V_mean_old = V_mean.copy()
        V_env_old = V_env.copy()
        Q_env = np.zeros([env.state_space.n, n_env_actions])
        iterations += 1
        # sample(range(env.state_space.n), env.state_space.n)
        # range(env.state_space.n)
        for state in range(env.state_space.n):
            if (
                state < env.width
                or state >= env.state_space.n - env.width
                or state % env.width == 0
                or state + 1 % env.width == 0
            ):
                continue
            for env_action in sample(range(n_env_actions), n_env_actions):
                # get current environment
                current_walls = walls.copy()
                right_coord = scalar_to_coordinate(env, state + 1)
                bot_coord = scalar_to_coordinate(env, state + env.width)
                if env_action == 0:
                    if right_coord in current_walls:
                        current_walls.remove(right_coord)
                    if bot_coord in current_walls:
                        current_walls.remove(bot_coord)
                elif env_action == 1:
                    if (
                        current_env.grid.get(*right_coord) is None
                        and env.agent_start_pos != right_coord
                    ):
                        current_walls.append(right_coord)
                    if bot_coord in current_walls:
                        current_walls.remove(bot_coord)
                elif env_action == 2:
                    if (
                        current_env.grid.get(*bot_coord) is None
                        and env.agent_start_pos != bot_coord
                    ):
                        current_walls.append(bot_coord)
                    if right_coord in current_walls:
                        current_walls.remove(right_coord)
                elif env_action == 3:
                    if (
                        current_env.grid.get(*right_coord) is None
                        and env.agent_start_pos != right_coord
                    ):
                        current_walls.append(right_coord)
                    if (
                        current_env.grid.get(*bot_coord) is None
                        and env.agent_start_pos != bot_coord
                    ):
                        current_walls.append(bot_coord)
                current_env = ConstructedMazeEnv(size=env.width, walls=current_walls)
                # inside max for sampled rewards
                for reward_index in range(n_rewards):
                    temp_Q_rew = np.zeros(env.action_space.n)
                    for action in range(env.action_space.n):
                        temp_Q_rew[action] = env.gamma * np.dot(
                            current_env.get_transition_probabilities(state, action),
                            V[reward_index, :],
                        )
                    Q_env[state, env_action] += max(temp_Q_rew) / n_rewards
                # mean reward function
                temp_Q_rew = np.zeros(env.action_space.n)
                for action in range(env.action_space.n):
                    temp_Q_rew[action] = env.gamma * np.dot(
                        current_env.get_transition_probabilities(state, action),
                        V_mean[:],
                    )
                Q_env[state, env_action] -= max(
                    temp_Q_rew
                )  # WE CAN ADD A FACTOR < 1 HERE TO PROMOTE SOLVABLE ENVIRONMENTS
            # get value for environment
            V_env[state] = max(Q_env[state, :])
            pol_env[state] = np.argmax(Q_env[state, :])

            # update walls based on environment policy in state
            right_coord = scalar_to_coordinate(env, state + 1)
            bot_coord = scalar_to_coordinate(env, state + env.width)
            if pol_env[state] == 0:
                if right_coord in walls:
                    walls.remove(right_coord)
                if bot_coord in walls:
                    walls.remove(bot_coord)
            elif pol_env[state] == 1:
                if (
                    current_env.grid.get(*right_coord) is None
                    and env.agent_start_pos != right_coord
                ):
                    walls.append(right_coord)
                if bot_coord in walls:
                    walls.remove(bot_coord)
            elif pol_env[state] == 2:
                if (
                    current_env.grid.get(*bot_coord) is None
                    and env.agent_start_pos != bot_coord
                ):
                    walls.append(bot_coord)
                if right_coord in walls:
                    walls.remove(right_coord)
            elif pol_env[state] == 3:
                if (
                    current_env.grid.get(*right_coord) is None
                    and env.agent_start_pos != right_coord
                ):
                    walls.append(right_coord)
                if (
                    current_env.grid.get(*bot_coord) is None
                    and env.agent_start_pos != bot_coord
                ):
                    walls.append(bot_coord)

            current_env = ConstructedMazeEnv(size=env.width, walls=walls)
            # update value functions
            for reward_index in range(n_rewards):
                for a in range(env.action_space.n):
                    Q[reward_index, state, a] = s_reward[reward_index][
                        state
                    ] + env.gamma * np.dot(
                        current_env.get_transition_probabilities(state, a),
                        V[reward_index, :],
                    )
                V[reward_index, state] = max(Q[reward_index, state, :])
            # mean reward function
            for a in range(env.action_space.n):
                Q_mean[state, a] = m_reward[state] + env.gamma * np.dot(
                    current_env.get_transition_probabilities(state, a), V_mean[:]
                )
            V_mean[state] = max(Q_mean[state, :])

        if (
            np.max(np.abs(V_old - V)) < delta
            and max(np.abs(V_mean_old - V_mean)) < delta
            and max(np.abs(V_env_old - V_env)) < delta
        ):
            print("Max-Min Maze VI Iterations", iterations)
            break
        if iterations % 10 == 0:
            print("Iterations", iterations)
            print("Value wrt Mean", V_mean[env.width + 1])
            print("Value wrt a sampled rew ", V[0, env.width + 1])
            print("Env Value", V_env[env.width + 1])
            it_env = ConstructedMazeEnv(size=env.width, walls=walls)
            save_initial_render(it_env, iterations)
        if iterations > 99:
            break
        curr_regret = evaluate_regret_of_maze(env, walls, s_reward, m_reward)
        if curr_regret > max_regret:
            max_regret = curr_regret
            max_walls = walls
        print(
            "Current Iteration",
            iterations,
            "Walls: Regret",
            curr_regret,
            " ---- Max Regret:",
            max_regret,
            "Max Walls",
            max_walls,
        )
    return max_walls
