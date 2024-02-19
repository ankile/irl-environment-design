import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from .constants import ParamTuple, gamma_limits, p_limits, R_limits, StateTransition
from .make_environment import Environment
from .inference.likelihood import expert_trajectory_log_likelihood


def plot_trajectories(N, M, trajectories, reward_matrix, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(M, N))

    # Plot rewards as heatmap
    im = ax.imshow(
        reward_matrix.reshape(N, M), cmap="turbo", origin="upper")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

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
    # To also move the x-axis label if you have one
    ax.xaxis.set_label_position("top")
    ax.set_title(f"{len(trajectories)} Expert Trajectories")


from matplotlib.patches import Rectangle


def plot_environment(reward_function, wall_states, start_state=(0, 0), ax=None,
                     annotate=True,
                     colorbar=False):
    # Assume the reward function is already reshaped to a 2D grid
    N, M = reward_function.shape
    # Identify wall states is the indixes into the

    wall_states = set([(s // M, s % M) for s in wall_states])

    if ax is None:
        fig, ax = plt.subplots()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.matshow(reward_function, cmap="turbo")

    # Annotate each cell with the reward, start, and wall
    for (i, j), val in np.ndenumerate(reward_function):
        if (i, j) == start_state:
            ax.text(j, i, "S", va="center", ha="center", color="black")
        elif (i, j) in wall_states:
            # Add a dark gray rectangle to represent the wall
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="darkgray"))
        elif val != 0 and annotate:
            ax.text(j, i, f"{round(val, 1):g}", va="center", ha="center", color="white")
    ax.text(0,0, "S", va="center", ha="center", color="white")
    ax.set_title(f"Environment")






def plot_environments_with_regret(envs):
    N, M, R = envs[0].N, envs[0].M, envs[0].R_sample_mean

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
        if env.trajectories is not None:
            plot_trajectories(N, M, env.trajectories, R, ax=ax2)

            # Remove the colorbar from the second plot
            ax2.get_images()[0].colorbar.remove()

            fig.suptitle(
                f"Regret: {env.regret:.3f}, Log regret: {env.log_regret:.3f}\n({env.likelihoods} / {env.log_likelihoods})"
            )

        else:
            fig.suptitle(f"Regret: {env.regret:.3f}")


def plot_posterior_distribution(
    posterior_samples: list[ParamTuple], 
    N: int, 
    M: int, 
    absorbing_states: np.array=None,
    true_params: ParamTuple = None, 
    ax=None
):
    """
    Plot the join distribution of p and gamma from the posterior samples as a 2D histogram.
    Plot the mean of the reward distribution as a grid
    """
    if ax is None:
        fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
    # Unzipping the list of tuples
    p_values, gamma_values, R_values = zip(*(posterior_samples))

    # Plotting the 2D distribution
    axs[0].scatter(p_values, gamma_values, alpha=0.3)
    axs[0].set_title("Posterior distribution over $\gamma$ and $p$")
    axs[0].set_xlabel("$p_i$")
    axs[0].set_ylabel("$\\gamma_i$")
    axs[0].grid(True)
    axs[0].set_xlim(p_limits)
    axs[0].set_ylim(gamma_limits)
    if true_params is not None:
            axs[0].scatter(
                true_params.p,
                true_params.gamma,
                marker="*",
                color="red",
                label="True parameters",
            )
            axs[0].legend()

    posterior_samples_reward_mean = np.mean(R_values, axis = 0)
    posterior_samples_reward_variance = np.var(R_values, axis=0)
    posterior_samples_reward_mean = posterior_samples_reward_mean.reshape(N,M)
    posterior_samples_reward_variance = posterior_samples_reward_variance.reshape(N,M)


    #plot mean of estimated rewards
    img_1 = axs[1].imshow(posterior_samples_reward_mean, cmap=plt.cm.seismic, vmin=np.min(R_values), vmax=np.max(R_values))
    if not true_params:
        plt.colorbar(img_1, ax=axs[1])
    axs[1].set_title("Mean of Reward Samples")


    if absorbing_states is not None:
        N_goal, M_goal = absorbing_states[0] // N, absorbing_states[0] % N
        axs[1].add_patch(Circle((M_goal, N_goal), 0.3, color="darkgray", label="Absorbing States"))
        for goal_state in absorbing_states[1:]:
            N_goal, M_goal = goal_state // N, goal_state % N
            axs[1].add_patch(Circle((M_goal, N_goal), 0.3, color="darkgray"))


    #plot the true rewards
    if true_params:
        img_2 = axs[2].imshow(np.reshape(true_params.R, (N,M)), cmap=plt.cm.seismic, vmin = 0)
        plt.colorbar(img_2, ax=axs[2])
        axs[2].set_title("True Rewards")
        
        if absorbing_states is not None:
            N_goal, M_goal = absorbing_states[0] // N, absorbing_states[0] % N
            axs[2].add_patch(Circle((M_goal, N_goal), 0.3, color="darkgray", label="Absorbing States"))
            for goal_state in absorbing_states[1:]:
                N_goal, M_goal = goal_state // N, goal_state % N
                axs[2].add_patch(Circle((M_goal, N_goal), 0.3, color="darkgray"))


def mcmc_diagnostics(samples: list[ParamTuple], true_params: ParamTuple = None):

    samples_p = [sample[0] for sample in samples]
    samples_gamma = [sample[1] for sample in samples]

    fig, axs = plt.subplots(2,2, figsize = (15,10))
    axs[0,0].plot(samples_p)
    axs[0,0].set_title("Traceplot p, all iterations")
    axs[0,0].set_xlabel("Iterations")
    axs[0,0].axhline(true_params.p, label = "True $p$", c="green")


    axs[0,1].plot(samples_gamma)
    axs[0,1].set_title("Traceplot $\gamma$, all iterations")
    axs[0,1].set_xlabel("Iterations")
    axs[0,1].axhline(true_params.gamma, label = "True $\gamma$", c="green")

    axs[1,0].set_title("Autocorrelation $p$")
    pd.plotting.autocorrelation_plot(samples_p, ax=axs[1,0])

    axs[1,1].set_title("Autocorrelation $\gamma$")
    pd.plotting.autocorrelation_plot(samples_gamma, ax=axs[1,1])

    fig.legend(loc="upper right", fancybox=True, shadow=True)




def plot_log_likelihood(param_values: ParamTuple, 
                        expert_trajectories: list[tuple[Environment, list[StateTransition]]], 
                        goal_states: list,
                        ax = None
                        ):

    '''
    Plots the posterior distribution of $p$ and $\gamma$ holding R constant.

    Args:
    --param_values, ParamTuple: Containing the values for the posterior
    --expert trajectories: list of tuples containing environments and their respective expert environments
    --goal_states: list of goal states, flattened
    '''
    if ax is None:
        _, axs = plt.subplots()
    else:
        axs = ax
    n_samples_per_axis = 15

    gammas = np.linspace(0.5, 0.95, n_samples_per_axis)
    ps = np.linspace(0.95, 0.5, n_samples_per_axis)

    likelihoods = np.zeros(shape = (n_samples_per_axis, n_samples_per_axis))

    for idx_p, p in enumerate(ps):
        for idx_gamma, gamma in enumerate(gammas): #only up to 0.95 because it gets unstable for higher gamma

            proposed_parameter = ParamTuple(p=p, gamma=gamma, R=param_values.R)

            likelihood = expert_trajectory_log_likelihood(
                proposed_parameter, expert_trajectories, goal_states
            )
            likelihoods[idx_p, idx_gamma] = likelihood

    index_p_true = (np. abs(ps - param_values.p)). argmin()
    index_gamma_true = (np. abs(gammas - param_values.gamma)). argmin()

    im = axs.imshow(
        likelihoods, cmap="viridis", origin="upper"
    )
    plt.colorbar(im, orientation="vertical")
    #Here we need to transpose everything as imshow transposes the image.
    axs.set_ylabel("p")
    axs.set_xlabel("$\gamma$")
    axs.set_yticks(np.arange(n_samples_per_axis), np.round(ps, 2))
    axs.set_xticks(np.arange(n_samples_per_axis), np.round(gammas, 2), rotation='vertical')
    axs.set_title(f"Posterior over $p$ and $\gamma$\nlog likelihood.")
    plt.plot(index_gamma_true, index_p_true, "og", label = "True Values")
    axs.legend()