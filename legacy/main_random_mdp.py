import numpy as np

from random_mdp import RandomMDP
from helper import *
from multi_env_birl import bayesian_reward_learning
from environment_design import *
import copy
from random import sample
import matplotlib.pyplot as plt


prefix = 'birl_'
n_states = 40
n_actions = 4
rad_demo = 0.5
rad_test = 0.75
n_demo = 20
n_test = 250

birl_sample_size = 5000
ed_reward_samples = 300
iterations = 5

n_episodes = 10
traj_length = 10
obs_per_eps = 1


avg_utility_ed = [0 for _ in range(n_episodes)]
avg_utility_rand = [0 for _ in range(n_episodes)]
avg_utility_fixed = [0 for _ in range(n_episodes)]
avg_utility_opt = [0 for _ in range(n_episodes)]


avg_ut_ed = [0 for _ in range(11)]
avg_ut_rand = [0 for _ in range(11)]
avg_ut_fixed = [0 for _ in range(11)]
avg_ut_opt = [0 for _ in range(11)]

posterior_mean_ed = np.random.randint(1, 10, size=n_states) / 10
posterior_mean_rand = np.random.randint(1, 10, size=n_states) / 10
posterior_mean_f = np.random.randint(1, 10, size=n_states) / 10

utility_opt = 0
for i in range(iterations):
    print("Iteration", i)
    observations_rand = []
    observations_fixed = []
    observations_ed = []
    utility_rand = []
    utility_fixed = []
    utility_ed = []

    mdp = RandomMDP(n_states=n_states, n_actions=n_actions, n_demo=n_demo, n_test=n_test, rad_demo=rad_demo, rad_test=rad_test)

    ed_mdp = copy.deepcopy(mdp)
    rand_mdp = copy.deepcopy(mdp)

    original_rewards = mdp.get_rewards()

    opt_values = evaluate_reward(mdp, original_rewards)
    utility_opt += opt_values / iterations
    print("opt values", opt_values)

    ed_mdp = copy.deepcopy(mdp)
    rand_mdp = copy.deepcopy(mdp)

    for episode in range(n_episodes):
        if episode > 0:
            rand_mdp = domain_randomisation(mdp)
            # for s in range(mdp.state_space.n):
            #     for a in range(mdp.action_space.n):
            #         assert sum(np.abs(rand_mdp.P[s, a, :] - mdp.P[s, a, :])) <= mdp.rad_demo
            # Maximin Bayesian Regret Env
            s_reward = sample(posterior_samples_ed[-1500:], ed_reward_samples)
            m_reward = sum(s_reward) / len(s_reward)
            ed_mdp = extended_value_iteration(mdp, m_reward, s_reward)

        if episode == 0:
            for _ in range(2):
                obs = get_expert_trajectory(mdp, traj_length)
                observations_rand.append(obs)
                observations_fixed.append(obs)
                observations_ed.append(obs)
        else:
            observations_ed.append(get_expert_trajectory(ed_mdp, traj_length))
            observations_rand.append(get_expert_trajectory(rand_mdp, traj_length))
            observations_fixed.append(get_expert_trajectory(mdp, traj_length))
        posterior_samples_ed, posterior_mean_ed, posterior_map_ed, posterior_std_ed = bayesian_reward_learning(mdp, observations_ed, birl_sample_size, last_reward=posterior_mean_ed, proposal_distr='grid')
        posterior_samples_rand, posterior_mean_rand, posterior_map_rand, posterior_std_rand = bayesian_reward_learning(mdp, observations_rand, birl_sample_size, last_reward=posterior_mean_rand, proposal_distr='grid')
        posterior_samples_f, posterior_mean_f, posterior_map_f, posterior_std_f = bayesian_reward_learning(mdp, observations_fixed, birl_sample_size, last_reward=posterior_mean_f, proposal_distr='grid')

        if episode == 0:
            mdp.set_rewards(original_rewards)
            v = evaluate_reward(mdp, posterior_mean_ed)
            utility_ed.append(v)
            utility_rand.append(v)
            utility_fixed.append(v)
        else:
            # evaluate rewards
            mdp.set_rewards(original_rewards)
            utility_ed.append(evaluate_reward(mdp, posterior_mean_ed))
            utility_rand.append(evaluate_reward(mdp, posterior_mean_rand))
            utility_fixed.append(evaluate_reward(mdp, posterior_mean_f))

        print('Fixed', utility_fixed)
        print('Dom. Rand.', utility_rand)
        print('ED-BIRL', utility_ed)

    # evaluate on different amounts of change
    ut_ed = []
    ut_fixed = []
    ut_rand = []
    ut_opt = []
    p_range = np.linspace(0, 1, num=11)*1.5
    for r_test in p_range:
        print("r_test", r_test)
        mdp.rad_test = r_test
        mdp.update_test_env()
        mdp.set_rewards(original_rewards)
        ut_ed.append(evaluate_reward(mdp, posterior_mean_ed))
        ut_rand.append(evaluate_reward(mdp, posterior_mean_rand))
        ut_fixed.append(evaluate_reward(mdp, posterior_mean_f))
        ut_opt.append(evaluate_reward(mdp, original_rewards))

    avg_utility_ed = [avg_utility_ed[i] + utility_ed[i]/iterations for i in range(n_episodes)]
    avg_utility_rand = [avg_utility_rand[i] + utility_rand[i]/iterations for i in range(n_episodes)]
    avg_utility_fixed = [avg_utility_fixed[i] + utility_fixed[i]/iterations for i in range(n_episodes)]

    avg_ut_ed = [avg_ut_ed[i] + ut_ed[i]/iterations for i in range(11)]
    avg_ut_rand = [avg_ut_rand[i] + ut_rand[i]/iterations for i in range(11)]
    avg_ut_fixed = [avg_ut_fixed[i] + ut_fixed[i]/iterations for i in range(11)]
    avg_ut_opt = [avg_ut_opt[i] + ut_opt[i]/iterations for i in range(11)]

# save values: for the specific run with radius = 0.5
np.savetxt('plots' + '/' + prefix + "avg_utility_ed.txt", avg_utility_ed)
np.savetxt('plots' + '/' + prefix + "avg_utility_fixed.txt", avg_utility_fixed)
np.savetxt('plots' + '/' + prefix + "avg_utility_rand.txt", avg_utility_rand)
np.savetxt('plots' + '/' + prefix + "avg_opt_utility.txt", np.array([utility_opt]))


np.savetxt('plots' + '/' + prefix + "avg_ut_ed.txt", avg_ut_ed)
np.savetxt('plots' + '/' + prefix + "avg_ut_fixed.txt", avg_ut_fixed)
np.savetxt('plots' + '/' + prefix + "avg_ut_rand.txt", avg_ut_rand)
np.savetxt('plots' + '/' + prefix + "avg_ut_opt.txt", avg_ut_opt)


# plotting
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), avg_utility_ed, label="ED-BIRL")
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), avg_utility_rand, label="Domain Randomisation")
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), avg_utility_fixed, label="Fixed Environment IRL")
plt.legend(fontsize=14)
plt.xlabel("Round", fontsize=14)
plt.ylabel("Utility", fontsize=14)
plt.savefig('plots' + '/' + prefix + "Utility_Progression.png", bbox_inches='tight', dpi=300)
plt.close()


plt.plot(p_range/1.5, avg_ut_ed, label="ED-BIRL")
plt.plot(p_range/1.5, avg_ut_rand, label="Domain Randomisation")
plt.plot(p_range/1.5, avg_ut_fixed, label="Fixed Environment IRL")
plt.legend(fontsize=14)
plt.xlabel("Amount of Variation in Transitions", fontsize=14)
plt.ylabel("Utility", fontsize=14)
plt.savefig('plots' + '/' + prefix + "Transition_Variation_Plot.png", bbox_inches='tight', dpi=300)
plt.close()


# plotting
opt = np.ones([len(avg_utility_ed)])*utility_opt
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), opt - np.array(avg_utility_ed), label="ED-BIRL")
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), opt - np.array(avg_utility_rand), label="Domain Randomisation")
plt.plot(np.linspace(1, len(avg_utility_ed), num=len(avg_utility_ed)), opt - np.array(avg_utility_fixed), label="Fixed Environment IRL")
plt.legend(fontsize=14)
plt.xlabel("Round", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig('plots' + '/' + prefix + "Loss_Progression.png", bbox_inches='tight', dpi=300)
plt.close()


plt.plot(p_range/1.5, np.array(avg_ut_opt) - np.array(avg_ut_ed), label="ED-BIRL")
plt.plot(p_range/1.5, np.array(avg_ut_opt) - np.array(avg_ut_rand), label="Domain Randomisation")
plt.plot(p_range/1.5, np.array(avg_ut_opt) - np.array(avg_ut_fixed), label="Fixed Environment IRL")
plt.legend(fontsize=14)
plt.xlabel("Amount of Variation in Transitions", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig('plots' + '/' + prefix + "Loss_Transition_Variation_Plot.png", bbox_inches='tight', dpi=300)
plt.close()