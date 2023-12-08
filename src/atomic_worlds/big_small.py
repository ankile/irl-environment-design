import os

import numpy as np
import seaborn as sns
from src.utils.enums import TransitionMode

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from src.visualization.worldviz import plot_world_reward

from src.worlds.mdp2d import Experiment_2D


def smallbig_reward(
    height: int,
    width: int,
    big_reward: float,
    small_reward_frac: float,
) -> dict:
    """
    Places the small and big rewards in the bottom-left and bottom-right corners, respectively.

    returns a dictionary of rewards for each state in the gridworld.
    """

    reward_dict = {}

    small_reward = big_reward * small_reward_frac

    # Put small reward in lower left corner
    small_reward_state = (height - 1) * width
    reward_dict[small_reward_state] = small_reward

    # Put big reward in lower right corner
    big_reward_state = height * width - 1
    reward_dict[big_reward_state] = big_reward

    return reward_dict


def make_smallbig_experiment(
    height: int,
    width: int,
    big_reward: float,
    small_reward_frac: float,
) -> Experiment_2D:
    wall_dict = smallbig_reward(
        height,
        width,
        big_reward,
        small_reward_frac,
    )

    experiment = Experiment_2D(
        height,
        width,
        rewards_dict=wall_dict,
        transition_mode=TransitionMode.SIMPLE,
    )

    return experiment


if __name__ == "__main__":
    params = {
        "height": 7,
        "width": 7,
        "big_reward": 300,
        "small_reward_frac": 0.5,
    }

    experiment = make_smallbig_experiment(**params)

    experiment.set_user_params(
        prob=0.99,
        gamma=0.9,
        params=params,
        transition_func=lambda T, height, width, prob, params: T,
    )

    # Make plot with 5 columns where the first column is the parameters
    # and the two plots span two columns each

    # create figure with 5 columns
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 5, figure=fig)

    # add text to first column
    ax1 = fig.add_subplot(gs[0, 0])  # type: ignore
    ax1.axis("off")

    # add subplots to remaining 4 columns
    ax2 = fig.add_subplot(gs[0, 1:3])  # type: ignore
    ax3 = fig.add_subplot(gs[0, 3:5])  # type: ignore

    # Adjust layout and spacing (make room for titles)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Add the parameters to the first subplot
    ax1.text(
        0.05,
        0.95,
        "\n".join([f"{k}: {v}" for k, v in params.items()]),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax1.transAxes,
    )

    # Have no mask for now
    mask = None

    plot_world_reward(experiment, setup_name="BigSmall", ax=ax2, show=False, mask=mask)

    experiment.mdp.solve(
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        heatmap_mask=mask,
        base_dir="local_images",
        label_precision=1,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()
