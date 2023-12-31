import random
import copy

from tqdm import tqdm
from maze_env import ConstructedMazeEnv
from auxiliary.auxiliary import *
from auxiliary.mdp_solver import *
from random import sample  # random order in for-loops of VI
from multi_env_birl import get_likelihood, make_obs_compact, get_log_likelihood
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
    for _ in range(int(size / 3)):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.1))
    for _ in range(int(size / 3)):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.2))
    for _ in range(int(size / 3)):
        candidates.append(non_blocking_domain_randomisation(env, prob=0.3))
    return candidates


# Evaluate Bayesian regret of environment w.r.t. sampled rewards and posterior mean
def evaluate_regret_of_maze(env, walls, s_reward, m_reward):
    regret = 0
    maze = ConstructedMazeEnv(
        size=env.width, walls=walls
    )  # just change the transition function, maybe
    # Calculate V^\pi for our point estimate of the reward

    maze.rewards = m_reward
    V_mean, x, y = value_iteration(maze)

    negatives = 0

    # Calculate V^* for each sampled reward
    for reward in s_reward:
        maze.rewards = reward
        V, x, y = value_iteration(maze)
        negatives += int((V[env.width + 1] - V_mean[env.width + 1]) < 0)

        regret += V[env.width + 1] / len(s_reward)

    regret -= V_mean[env.width + 1]
    # maze.render()
    # time.sleep(2)
    # maze.reset()
    # maze.close()
    print(f"Regret: {regret}, Negatives pct: {negatives / len(s_reward):.5%}")
    return regret


# Evaluate Bayesian regret of environment w.r.t. sampled rewards and posterior mean
def evaluate_likelihood_regret_of_maze(env, walls, s_reward, m_reward, compact_obs):
    regret = 0
    maze = ConstructedMazeEnv(
        size=env.width, walls=walls
    )  # just change the transition function, maybe

    # Calculate l^\pi for our point estimate of the reward
    # likelihood_pi = get_log_likelihood(
    #     m_reward,
    #     compact_obs,
    # )

    likelihood_pi = get_likelihood(
        m_reward,
        compact_obs,
    )[0]

    # Calculate l^* for each sampled reward
    for reward in s_reward:
        # likelihood_star += get_log_likelihood(
        #     reward,
        #     compact_obs,
        # )
        likelihood_star = get_likelihood(
            reward,
            compact_obs,
        )[
            0
        ] / len(s_reward)

        regret += np.abs(likelihood_star - likelihood_pi)

    # maze.render()
    # time.sleep(2)
    # maze.reset()
    # maze.close()
    return regret


# brute force
def brute_force_maze_design(env, s_reward, m_reward, candidate_size):
    candidate_walls = get_environment_candidates(env, candidate_size)
    bayes_regret = []
    it = tqdm(candidate_walls, desc="Candidate", leave=False)
    for idx, walls in enumerate(it):
        bayes_regret.append(evaluate_regret_of_maze(env, walls, s_reward, m_reward))
        if idx % 10 == 0:
            it.set_postfix({"Maze Design: Candidates No.": idx})
    return candidate_walls, bayes_regret


# brute force
def brute_force_maze_design_likelihood(
    env, s_reward, m_reward, candidate_size, observations
):
    compact_obs = make_obs_compact(observations)
    candidate_walls = get_environment_candidates(env, candidate_size)
    bayes_regret = []
    it = tqdm(candidate_walls, desc="Candidate", leave=False)
    for idx, walls in enumerate(it):
        bayes_regret.append(
            evaluate_likelihood_regret_of_maze(
                env, walls, s_reward, m_reward, compact_obs
            )
        )
    return candidate_walls, bayes_regret


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
