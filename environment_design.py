import copy

import numpy as np
from auxiliary import mdp_solver


""" Domain Randomisation """

def domain_randomisation(mdp):
    temp_mdp = copy.deepcopy(mdp)
    for state in range(mdp.state_space.n):
        state_transition_idx = np.random.choice(range(mdp.n_demo))
        temp_mdp.set_state_transition_probabilities(state, mdp.demo_env[state][state_transition_idx])
    return temp_mdp


""" Extended Value Iteration """


def extended_value_iteration(mdp, m_reward, s_reward, delta=0.01):
    n_rewards = len(s_reward)
    n_env_actions = mdp.n_demo

    # value functions w.r.t. sampled rewards
    V = np.zeros([n_rewards, mdp.state_space.n])
    Q = np.zeros([n_rewards, mdp.state_space.n, mdp.action_space.n])

    V_mean = np.zeros(mdp.state_space.n)
    Q_mean = np.zeros([mdp.state_space.n, mdp.action_space.n])

    temp_mdp = copy.deepcopy(mdp)

    # pre-train value functions
    for i in range(n_rewards):
        temp_mdp.rewards = s_reward[i]
        V[i], x, y = mdp_solver.value_iteration(temp_mdp)
    temp_mdp.rewards = m_reward
    V_mean, x, y = mdp_solver.value_iteration(temp_mdp)
    temp_mdp.rewards = mdp.get_rewards()  # reset rewards

    # value function for environment
    V_env = np.zeros(mdp.state_space.n)
    Q_env = np.zeros([mdp.state_space.n, n_env_actions])
    pol_env = np.zeros(mdp.state_space.n)

    max_regret = 0
    iterations = 0
    while True:
        V_old = V.copy()
        V_mean_old = V_mean.copy()
        V_env_old = V_env.copy()
        Q_env = np.zeros([temp_mdp.state_space.n, n_env_actions])  # has to be reset
        iterations += 1
        # sample(range(env.state_space.n), env.state_space.n)
        # range(env.state_space.n)
        for state in range(temp_mdp.state_space.n):

            for e_action in range(n_env_actions):
                # set transitions in state to some other transition matrix
                temp_mdp.set_state_transition_probabilities(state, temp_mdp.demo_env[state][e_action])
                for reward_index in range(n_rewards):  # for sampled rewards
                    temp_Q_rew = np.zeros(temp_mdp.action_space.n)  # temporary Q values for current state
                    for action in range(temp_mdp.action_space.n):
                        temp_Q_rew[action] = temp_mdp.gamma * np.dot(temp_mdp.get_transition_probabilities(state, action), V[reward_index, :])
                    Q_env[state, e_action] += max(temp_Q_rew) / n_rewards
                temp_Q_rew = np.zeros(temp_mdp.action_space.n)  # now mean reward fct
                for action in range(temp_mdp.action_space.n):
                    temp_Q_rew[action] = temp_mdp.gamma * np.dot(temp_mdp.get_transition_probabilities(state, action), V_mean[:])
                Q_env[state, e_action] -= max(temp_Q_rew)
            V_env[state] = max(Q_env[state, :])
            pol_env[state] = np.argmax(Q_env[state, :])  # get the best env action

            # update transitions according to the current best env action
            temp_mdp.set_state_transition_probabilities(state, temp_mdp.demo_env[state][int(pol_env[state])])

            # update value functions
            for reward_index in range(n_rewards):
                for a in range(temp_mdp.action_space.n):
                    Q[reward_index, state, a] = s_reward[reward_index][state] + temp_mdp.gamma * np.dot(temp_mdp.get_transition_probabilities(state, a), V[reward_index, :])
                V[reward_index, state] = max(Q[reward_index, state, :])
            # mean reward function
            for a in range(temp_mdp.action_space.n):
                Q_mean[state, a] = m_reward[state] + temp_mdp.gamma * np.dot(temp_mdp.get_transition_probabilities(state, a), V_mean[:])
            V_mean[state] = max(Q_mean[state, :])

        # check for convergence
        if np.max(np.abs(V_old - V)) < delta and max(np.abs(V_mean_old - V_mean)) < delta and max(np.abs(V_env_old - V_env)) < delta:
            print("Total Extended VI Iterations", iterations)
            break
        if iterations % 20 == 0:
            print("Iterations", iterations)
            print("Value wrt Mean", V_mean[0])
            print("Value wrt a sampled rew ", V[0, 0])
            print("Env Value", V_env[0])
        if iterations > 200:
            break
    return temp_mdp


""" Continuous Perturbations """


def continuous_permutations(mdp, s_rewards, learning_rate=0.1, radius=0.3):
    n_rewards = len(s_rewards)
    m_reward = sum(s_rewards) / n_rewards
    # sets of value functions and policies
    V = np.zeros([n_rewards, mdp.state_space.n])
    Q = np.zeros([n_rewards, mdp.state_space.n, mdp.action_space.n])
    pol = np.zeros([n_rewards, mdp.state_space.n, mdp.action_space.n])

    # value function w.r.t. mean rewards
    V_m = np.zeros(mdp.state_space.n)
    Q_m = np.zeros([mdp.state_space.n, mdp.action_space.n])
    pol_m = np.zeros([mdp.state_space.n, mdp.action_space.n])

    # pre-train on base mdp
    mdp_copy = copy.deepcopy(mdp)
    for i in range(n_rewards):
        mdp_copy.rewards = s_rewards[i]
        V[i], Q[i], pol[i] = mdp_solver.value_iteration(mdp_copy)
    mdp_copy.rewards = m_reward
    V_m, Q_m, pol_m = mdp_solver.value_iteration(mdp_copy)

    # permutations
    delta = np.zeros([mdp.state_space.n, mdp.action_space.n, mdp.state_space.n])

    for step in range(100):
        for s in range(mdp.state_space.n):
            for a in range(mdp.action_space.n):
                delta[s, a, :] = gradient_step(mdp, V, V_m, pol, pol_m, n_rewards, s, a, radius, learning_rate, delta[s, a, :])
        mdp_copy.P += delta.copy()

        for i in range(n_rewards):
            mdp_copy.rewards = s_rewards[i]
            V[i], Q[i], pol[i] = mdp_solver.value_iteration(mdp_copy)
        mdp_copy.rewards = m_reward
        V_m, Q_m, pol_m = mdp_solver.value_iteration(mdp_copy)
        print("Step", step, "Value of Permutation", np.sum(V) / n_rewards - sum(V_m))
    return mdp_copy


# delta_sa is np.array(mdp.state_space.n)
def gradient_step(mdp, V, V_m, pol, pol_m, n_rewards, state, action, radius, learning_rate, delta_sa):
    for next_state in range(mdp.state_space.n):
        partial_grad = 0
        for r in range(n_rewards):
            if pol[r, state, action] == 1:
                unit = np.zeros(mdp.state_space.n)
                unit[next_state] = 1
                partial_grad += np.dot(mdp.get_transition_probabilities(state, action) + unit, V[r, :])
        partial_grad /= n_rewards
        if pol_m[state, action] == 1:
            unit = np.zeros(mdp.state_space.n)
            unit[next_state] = 1
            partial_grad -= np.dot(mdp.get_transition_probabilities(state, action) + unit, V_m[:])
        delta_sa[next_state] += learning_rate * partial_grad
    # projection
    m_delta_sa = sum(delta_sa) / len(delta_sa)
    delta_sa = radius * (delta_sa - m_delta_sa) / sum(abs(delta_sa - m_delta_sa))
    return delta_sa


def evaluate_permutation(V, V_m, n_rewards):
    np.sum(V) / n_rewards - sum(V_m)
    return




# def random_mdp_shift(mdp, radius):
#     for state in range(mdp.state_space.n):
#         for action in range(mdp.action_space.n):
#             mdp.P[state, action, :] += random_shift_transitions(mdp, radius)
#             mdp.P[state, action, :] = mdp.P[state, action, :].clip(min=0, max=1)
#             mdp.P[state, action, :] = mdp.P[state, action, :] / sum(mdp.P[state, action, :])
#     return mdp
#
#
# def random_shift_transitions(mdp, variation=0.3):
#     delta = np.random.uniform(-1, 1, mdp.state_space.n)
#     delta = delta - sum(delta) / len(delta)
#     delta = variation * delta / sum(np.abs(delta))
#     return delta
