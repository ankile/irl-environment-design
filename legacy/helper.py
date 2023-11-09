import numpy as np
from auxiliary.mdp_solver import *
import copy


def get_expert_trajectory(env, length, beta=30):
    traj = []
    V, Q, pol = value_iteration(env)
    env.current_state = env.start_state
    for _ in range(length):
        # Boltzmann action selection
        Q_exp = np.exp(beta * Q)
        Q_boltz = Q_exp[env.current_state, :] / np.sum(Q_exp[env.current_state, :])
        action = np.random.choice(env.action_space.n, p=Q_boltz)
        # action = np.argmax(Q[state, :])
        traj.append([env.current_state, action])
        env.step(action)
    print(traj)
    obs = [env, traj]
    return obs


def evaluate_reward(mdp, reward_fct):
    ev_mdp = copy.deepcopy(mdp)
    utility = 0
    avg_utility = 0
    for t in range(mdp.n_test+1):
        ev_mdp.set_rewards(reward_fct)
        ev_mdp.set_transition_probabilities(mdp.test_env[t])
        tV, tQ, policy = value_iteration(ev_mdp)
        ev_mdp.set_rewards(mdp.get_rewards())
        V, Q = policy_evaluation(ev_mdp, policy)
        utility += V[0] / (mdp.n_test+1)
        avg_utility += sum(V) / (mdp.state_space.n * (mdp.n_test+1))
    return avg_utility  # utility, avg_utility
