import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
import copy


class RandomMDP(gym.Env):
    """
    Random MDP

    """

    def __init__(
            self,
            n_states=None,
            n_actions=None,
            n_demo=None,    # no of demo environments (per state)
            n_test=None,    # no of test environm
            rad_demo=None,    # l_infty distance for demo environments
            rad_test=None      # l_infty distance for test environments
    ):
        self.type = 'Random MDP'

        # # Action enumeration for this environment
        # self.actions = RandomMDP.Actions

        self.n_states = n_states
        self.n_actions = n_actions

        self.n_demo = n_demo
        self.n_test = n_test
        self.rad_demo = rad_demo
        self.rad_test = rad_test

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(n_actions)

        # State space. Enumeration of cells
        self.state_space = spaces.Discrete(n_states)

        # Initial state
        self.start_state = 0

        # Test and Demo Environments Initialisation
        self.test_env = []      # list of transition functions
        self.demo_env = [[] for _ in range(self.n_states)]

        # Current state
        self.current_state = self.start_state

        # Discount factor
        self.gamma = 0.9

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # State-only reward function.
        self.rewards = np.round(np.random.beta(0.45, 0.6, self.state_space.n), 1)

        # Transition function
        self.P = np.zeros([self.state_space.n, self.action_space.n, self.state_space.n])

        for s in range(self.state_space.n):
            for a in range(self.action_space.n):
                param = np.ones(self.n_states) / (0.9*self.n_states)
                self.P[s, a] = np.random.dirichlet(param)
        # round the probabilities to avoid floating point errors
        self.P = np.round(self.P, 5)
        for s in range(self.state_space.n):
            for a in range(self.action_space.n):
                summed = 1 - sum(self.P[s, a, :])
                if summed != 0:
                    indices = np.where((0 < self.P[s, a, :] + summed) & (self.P[s, a, :] + summed < 1))[0]
                    while True:
                        index = np.random.choice(indices)
                        if 0 < self.P[s, a, index] + summed < 1:
                            self.P[s, a, index] += summed
                            break
                        else:
                            print("ERROR when defining transition probabilities")


        # Generate Test Environments
        self.test_env.append(self.P)    # add base transitions
        for _ in range(self.n_test):
            self.test_env.append(self.generate_test_env())

        # Generate Demo Environments
        for state in range(self.state_space.n):
            for _ in range(1):
                self.demo_env[state].append(self.P[state, :, :])    # add base transitions
            for _ in range(self.n_demo):
                self.demo_env[state].append(self.generate_demo_env(state))

    # Generate Test Environments
    def update_test_env(self):
        # reset test environments
        self.test_env = []
        # generate test environments
        self.test_env.append(self.P)    # add base transitions
        for _ in range(self.n_test):
            self.test_env.append(self.generate_test_env())

    # Get transition probabilities
    def get_transition_probabilities(self, state, action):
        return self.P[state, action, :]

    # Get reward function
    def get_reward(self, state_index):
        return self.rewards[state_index]

    def get_rewards(self):
        return self.rewards

    def set_rewards(self, rewards):
        self.rewards = rewards

    def step(self, action):
        self.current_state = np.random.choice(range(self.state_space.n), p=self.P[self.current_state, action, :])

    def generate_test_env(self):
        test_P = np.zeros([self.state_space.n, self.action_space.n, self.state_space.n])
        for state in range(self.state_space.n):
            test_P[state, :, :] = self.generate_perturbed_state_transition_matrix(self.P[state, :, :], self.rad_test)
        return test_P
        # test_P = copy.deepcopy(self.P)
        # for state in range(self.state_space.n):
        #     for action in range(self.action_space.n):
        #         test_P[state, action, :] += self.random_shift(self.rad_test)
        #         test_P[state, action, :] = test_P[state, action, :].clip(min=0, max=1)
        #         test_P[state, action, :] = test_P[state, action, :] / sum(test_P[state, action, :])
        # return test_P

    def generate_demo_env(self, state):
        return self.generate_perturbed_state_transition_matrix(self.P[state, :, :], self.rad_demo)
        # demo_state_P = copy.deepcopy(self.P[state, :, :])
        # for action in range(self.action_space.n):
        #     demo_state_P[action, :] += self.random_shift(self.rad_demo)
        #     demo_state_P[action, :] = demo_state_P[action, :].clip(min=0, max=1)
        #     demo_state_P[action, :] = demo_state_P[action, :] / sum(demo_state_P[action, :])
        # return demo_state_P

    # # returns random transition shift (for a state-action pair) with at most "radius" difference
    # def random_shift(self, radius):
    #     p = self.state_space.n/1000
    #     a = np.random.choice([-2, 0, 2], self.state_space.n, p=[p, 1-2*p, p])
    #     delta = np.random.normal(a, 1, self.state_space.n)
    #     # delta = np.random.uniform(-1, 1, self.state_space.n)
    #     delta = delta - sum(delta) / len(delta)         # shift sums to 0
    #     if sum(np.abs(delta) > 0):
    #         delta = radius * delta / sum(np.abs(delta))     # shift at most radius
    #     # print("delta", np.round(delta, 2))
    #     return delta

    # set state-transition matrix to some value
    def set_state_transition_probabilities(self, state, state_P):
        self.P[state, :, :] = state_P

    def set_transition_probabilities(self, new_P):
        self.P = copy.deepcopy(new_P)

    # og_state_P is a state-transition matrix
    def generate_perturbed_state_transition_matrix(self, og_state_P, radius):
        if radius == 0:
            return og_state_P
        state_P = np.zeros([self.action_space.n, self.state_space.n])
        for a in range(self.action_space.n):
            dist = 0
            counter = 0
            while max(0.01, radius-0.3) > dist or dist > radius + 0.15:    # < radius / 2 or dist > radius:
                state_P[a, :] = np.random.dirichlet((5 / radius) * og_state_P[a, :] + 0.1)

                state_P[a, :] = np.round(state_P[a, :], 5)
                summed = 1 - sum(state_P[a, :])
                if summed != 0:
                    indices = np.where((0 < state_P[a, :] + summed) & (state_P[a, :] + summed < 1))[0]
                    while True:
                        index = np.random.choice(indices)
                        if 0 < state_P[a, index] + summed < 1:
                            state_P[a, index] += summed
                            break
                        else:
                            print("ERROR when defining transition probabilities")

                dist = abs(state_P[a, :] - og_state_P[a, :]).sum()
                counter += 1
        #     # print(counter)
        # state_P = np.round(state_P, 5)
        # for a in range(self.action_space.n):
        #     summed = 1 - sum(state_P[a, :])
        #     if summed != 0:
        #         indices = np.where((0 < state_P[a, :] + summed) & (state_P[a, :] + summed < 1))[0]
        #         while True:
        #             index = np.random.choice(indices)
        #             if 0 < state_P[a, index] + summed < 1:
        #                 state_P[a, index] += summed
        #                 break
        #             else:
        #                 print("ERROR when defining transition probabilities")
        #         # if np.sum(state_P[a, :]) != 1:
        #             # print("....", print(np.sum(state_P[a, :])))
        #             # print(np.random.choice(range(40), p=state_P[a, :]))
        return state_P
