import math
from auxiliary.mdp_solver import *
import time
import seaborn as sns
import matplotlib.pylab as plt

folder_no = ""


# translate 2-dim coordinates to scalar states
def coordinate_to_scalar(env, coord):
    return env.width * coord[1] + coord[0]


# translate scalar states to 2-dim coordinates
def scalar_to_coordinate(env, state_index):
    return state_index % env.width, math.floor(state_index / env.width)


# get an expert trajectory leading to a goal states
def get_expert_trajectory(env):
    env.reset()
    max_steps = 2 * env.width
    traj = []
    V, Q, pol = value_iteration(env)
    for _ in range(max_steps):
        env.render()
        state = coordinate_to_scalar(env, env.agent_pos)
        # Boltzmann action selection
        Q_exp = np.exp(30 * Q)
        Q_boltz = Q_exp[state, :] / np.sum(Q_exp[state, :])
        action = np.random.choice(env.action_space.n, p=Q_boltz)
        # action = np.argmax(Q[state, :])
        obs, reward, done, info = env.step(action)
        traj.append([state, action])
        # traj.append([state, action])
        if done:
            env.render()
            # state = coordinate_to_scalar(env, env.agent_pos)
            # action = np.argmax(Q[state, :])
            # traj.append([state, action])
            time.sleep(1)
            print("Trajectory Finished")
            break
    print(traj)
    obs = [env, traj]
    # env.close()
    return obs


# plot reward and standard deviation heatmaps
def plot_heatmaps(posterior_mean, posterior_std, episode=0):
    size = np.sqrt(len(posterior_mean))

    # plot reward heatmap
    split_rewards = np.array_split(np.array(posterior_mean), size)
    ax = sns.heatmap(
        split_rewards,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_posterior_mean.png",
        bbox_inches="tight",
    )
    plt.close()

    # plot posterior std
    split_std = np.array_split(posterior_std, size)
    ax = sns.heatmap(
        split_std,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_posterior_std.png",
        bbox_inches="tight",
    )
    plt.close()

    # plot rounded reward heatmap
    posterior_mean_r = np.round(posterior_mean, decimals=1)
    split_rewards = np.array_split(np.array(posterior_mean_r), size)
    ax = sns.heatmap(
        split_rewards,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_rounded_mean.png",
        bbox_inches="tight",
    )
    plt.close()

    # plot scaled posterior std
    posterior_std_r = np.round(posterior_std, decimals=1)
    split_std = np.array_split(posterior_std_r, size)
    ax = sns.heatmap(
        split_std,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_rounded_std.png",
        bbox_inches="tight",
    )
    plt.close()

    # map posterior mean to (0,1)
    if max(posterior_mean) != min(posterior_mean):
        posterior_mean = [
            (item - min(posterior_mean)) / (max(posterior_mean) - min(posterior_mean))
            for item in posterior_mean
        ]

    # map posterior_std to (0,1)
    if max(posterior_std) != min(posterior_std):
        posterior_std = [
            (item - min(posterior_std)) / (max(posterior_std) - min(posterior_std))
            for item in list(posterior_std)
        ]

    # plot rounded reward heatmap
    posterior_mean = np.round(posterior_mean, decimals=1)
    split_rewards = np.array_split(np.array(posterior_mean), size)
    ax = sns.heatmap(
        split_rewards,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_rounded_scaled_mean.png",
        bbox_inches="tight",
    )
    plt.close()

    # plot scaled posterior std
    posterior_std = np.round(posterior_std, decimals=1)
    split_std = np.array_split(posterior_std, size)
    ax = sns.heatmap(
        split_std,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_rounded_scaled_std.png",
        bbox_inches="tight",
    )
    plt.close()


def save_initial_render(env, episode):
    initial_render = env.render(mode="rgb_array")
    plt.axis("off")
    plt.imshow(initial_render)
    plt.savefig(
        "plots" + folder_no + "/" + str(episode) + "_env.png", bbox_inches="tight"
    )
    plt.close()


# sample rewards from the uniform prior
def sample_from_uniform(env, sample_size):
    n_states = env.state_space.n
    samples = []
    for _ in range(sample_size):
        samples.append(np.random.uniform(0, 1, n_states))
    return samples
