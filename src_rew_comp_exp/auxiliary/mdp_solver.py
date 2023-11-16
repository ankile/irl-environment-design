import numpy as np
import math


# Implementation of standard value iteration to solve mazes
def value_iteration(env, delta=0.001):
    V = np.zeros(env.state_space.n)
    Q = np.zeros([env.state_space.n, env.action_space.n])
    pol = np.zeros([env.state_space.n, env.action_space.n])

    if env.type == 'Maze':
        eff_state_space = [i for i in range(env.state_space.n) if (i % env.width != 0) and (i % env.width != env.width - 1) and (i > env.width) and (i < env.width * (env.width - 1))]
    else:
        eff_state_space = range(env.state_space.n)
    counter = 0
    while True:
        counter += 1
        V_old = V.copy()
        for s in eff_state_space:
            for a in range(env.action_space.n):
                Q[s, a] = env.get_reward(s) + env.gamma * np.dot(env.get_transition_probabilities(s, a), V[:])
            V[s] = max(Q[s, :])
        if max(np.abs(V_old - V)) < delta:
            for s in eff_state_space:
                pol[s][np.argmax(Q[s, :])] = 1
            # print("No. Iterations:", counter)
            break
    return V, Q, pol


# policy evaluation
def policy_evaluation(env, policy, delta=0.0001):
    V = np.zeros(env.state_space.n)
    Q = np.zeros([env.state_space.n, env.action_space.n])
    while True:
        V_old = V.copy()
        for s in range(env.state_space.n):
            for a in range(env.action_space.n):
                Q[s, a] = policy[s, a] * (env.get_reward(s) + env.gamma * np.dot(env.get_transition_probabilities(s, a), V[:]))
            V[s] = np.sum(Q[s, :])
        if max(np.abs(V_old - V)) < delta:
            break
    return V, Q
