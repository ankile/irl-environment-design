from collections import namedtuple
import itertools
import os
import pickle
from datetime import datetime
from functools import partial

from typing import Callable, Dict, Tuple, cast
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.utils.multithreading import OptionalPool

from src.utils.policy import follow_policy, param_generator
from src.visualization.strategy import make_general_strategy_heatmap
from src.worlds.mdp2d import Experiment_2D
import src.worlds.mdp2d as mdp2d
from src.utils.optimization import soft_q_iteration
from src.utils.constants import beta_agent


ExperimentResult = namedtuple("ExperimentResult", ["data", "p2idx", "pidx2states"])



# def plot_bmap(
#     world: mdp2d.Experiment_2D,
#     parameter_mesh,
#     shaped_parameter_mesh,
#     # gammas: np.ndarray,
#     # probs: np.ndarray,
#     # start_state=0,
#     # ax=None,
#     # plot:bool = False
# ):
#     result = calculate_behavior_map(
#         experiment=world,
#         params=world.params,
#         # gammas=gammas,
#         # probs=probs,
#         paramterers=parameters,
#         # start_state=start_state,
#     )

#     # data = result.data

#     # if plot:

#     #     if ax is None:
#     #         _, ax = plt.subplots(figsize=(4, 4))

#     #     make_general_strategy_heatmap(
#     #         results=data,
#     #         probs=probs,
#     #         p2idx=None,
#     #         title=f"",
#     #         ax=ax,
#     #         gammas=gammas,
#     #         annot=False,
#     #         ax_labels=False,
#     #         num_ticks=5,
#     #         legend=False
#     #     )
#     return result


#TODO, only compute over Region of Interest.
def calculate_behavior_map(
    environment: mdp2d.Experiment_2D,
    reward_update: np.ndarray,
    parameter_mesh,
    shaped_parameter_mesh,
    # gammas: np.ndarray,
    # probs: np.ndarray,
) -> ExperimentResult:
    """
    Run an experiment with a given set of parameters and return the results.

    The results are:
    - data: a matrix of shape (len(probs), len(gammas)) where each entry is an index
            into the list of policies
    - p2idx: a dictionary mapping policies to indices
    """

    # data = np.zeros((len(probs), len(gammas)), dtype=np.int32)
    data: np.ndarray = np.zeros_like(shaped_parameter_mesh.flatten(), dtype=np.int32)
    p2idx: Dict[str, int] = {}
    pidx2states: Dict[list, int] = {}


    #Index for current policy, increased by 1 for each new policy.
    idx_policy = 0

    #Initialize policy, V, Q
    policy = None
    V = None
    Q = None


    # for (i, prob), (j, gamma) in itertools.product(enumerate(probs), enumerate(gammas)):
    for idx_parameter, parameter in enumerate(parameter_mesh):

        # experiment.set_user_params(
        #     prob=prob,
        #     gamma=gamma,
        #     use_pessimistic=False,
        # )
        # self.R = custom_reward_function(**parameter)[]

        # experiment.mdp.solve(
        #     save_heatmap=False,
        #     show_heatmap=False,
        #     heatmap_ax=None,
        #     heatmap_mask=None,
        #     label_precision=1,
        # )

        _transition_func = environment.transition_function(*parameter.T)
        _reward_func = environment.reward_function(*parameter.R)
        _gamma = parameter.gamma

        #Update the reward function with the maximum entropy reward update from the previous iteration.
        _reward_func += reward_update

        policy, Q, V = soft_q_iteration(
            _reward_func, _transition_func, gamma=_gamma, beta=beta_agent, return_what="all", Q_init=Q, V_init=V, policy_init=policy
        )

        #Convert stochastic Boltzmann policy into determinstic, greedy policy for rollouts.
        greedy_policy = np.argmax(policy, axis=1)
        greedy_policy = np.reshape(greedy_policy,  newshape=(environment.N, environment.M))


        policy_str, policy_states = follow_policy(
            greedy_policy,
            height=environment.N,
            width=environment.M,
            initial_state=environment.start_state,
            goal_states=environment.goal_states,
        )

        equivalent_policy_exists: bool = False
        
        if pidx2states == {}:
            #First iteration, no equivalent policies yet.
            p2idx[policy_str] = idx_policy
            pidx2states[idx_policy] = policy_states
            idx_policy += 1


        else:
        
            #Get all previous rollouts/ policies.
            policy_rollouts = pidx2states.values()

            #We initialize the equivalent policy as the current policy. If there exists an equivalent one, we later overwrite it.
            for policy_rollout in policy_rollouts:

                #Check if there exists an equivalent policy already. Here, we define equivalent as
                # two policies are equivalent if their rollouts are equal up to a permutation (in previous
                # versions, we defined two policies to be only equivalent if their rollouts are exactly the same).
                # We can test equality up to a permutation more efficiently by testing whether the policies have
                #the same length and whether the policies arrive in the same goal state.

                if (len(policy_rollout) == len(policy_states)) and (policy_rollout[-1] == policy_states[-1]):
                    # Check whether there exists an equivalent policy (up to permutation).
                    equivalent_policy_exists = True
                    equivalent_policy_rollout = policy_rollout

                    #Get index of equivalent policy.
                    equivalent_policy_rollout_idx = list(pidx2states.keys())[list(pidx2states.values()).index(equivalent_policy_rollout)]
                    break


            if not equivalent_policy_exists:
                #There exists no equivalent policy, so new policy index is created
                p2idx[policy_str] = idx_policy
                pidx2states[idx_policy] = policy_states
                idx_policy += 1

        #Update which policy sample (i,j) used.
        if equivalent_policy_exists:
            data[idx_parameter] = equivalent_policy_rollout_idx

        else:
            data[idx_parameter] = p2idx[policy_str]




    return ExperimentResult(data, p2idx, pidx2states)


def run_one_world(
    param_value: Tuple[str, float],
    *,
    create_experiment_func: Callable[..., Experiment_2D],
    transition_matrix_func: Callable[..., np.ndarray],
    default_params: dict,
    probs: np.ndarray,
    gammas: np.ndarray,
    get_start_state: Callable[[int, int], int],
    get_realized_probs_indices: Callable[[int, int], list] | None = None,
    get_goal_states: Callable[[int, int], set] | None = None,
):
    param, value = param_value
    params = {**default_params, param: value}
    experiment: Experiment_2D = create_experiment_func(**params)
    h, w = experiment.height, experiment.width

    realized_probs_indices = (
        get_realized_probs_indices(h, w)
        if get_realized_probs_indices is not None
        else None
    )

    goal_states = get_goal_states(h, w) if get_goal_states is not None else None

    data, p2idx, realized_probs = calculate_behavior_map(
        experiment,
        transition_matrix_func,
        params,
        gammas,
        probs,
        start_state=get_start_state(h, w),
        realized_probs_indices=realized_probs_indices,
        goal_states=goal_states,
    )
    return data, p2idx, param, value, realized_probs


def init_outdir(setup_name):
    setup_name = setup_name.replace(" ", "_").lower()
    output_dir = f"local_images/{setup_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def run_param_sweep(
    *,
    setup_name,
    default_params,
    search_parameters,
    create_experiment_func,
    transition_matrix_func,
    rows,
    cols,
    get_start_state,
    granularity,
    gammas,
    probs=None,
    scalers=None,  # When this is defined, that means we are using the scaled version, i.e., pessimism
    get_realized_probs_indices: Callable | None = None,
    get_goal_states: Callable[[int, int], set] | None = None,
    run_parallel=False,
    filename=None,
    save_metadata=False,
    show_plot=True,
    subtitle_location=0.95,
    p2idx_override=None,
    idx_map=None,
):
    assert not (
        probs is not None and scalers is not None
    ), "Cannot have both probs and scalers defined"

    assert (
        probs is not None or scalers is not None
    ), "Must have either probs or scalers defined"

    assert (
        get_realized_probs_indices is not None or scalers is None
    ), "Must have realized_prob_coord when scalers is defined"

    if probs is None:
        probs = cast(np.ndarray, scalers)

    run_one_world_partial = partial(
        run_one_world,
        create_experiment_func=create_experiment_func,
        transition_matrix_func=transition_matrix_func,
        default_params=default_params,
        probs=probs,
        gammas=gammas,
        get_start_state=get_start_state,
        get_realized_probs_indices=get_realized_probs_indices,
        get_goal_states=get_goal_states,
    )

    n_processes = (os.cpu_count() if run_parallel else 1) or 1
    with OptionalPool(processes=n_processes) as pool:
        strategy_data = list(
            tqdm(
                pool.imap(run_one_world_partial, param_generator(search_parameters)),
                total=rows * cols,
                desc=f"Running {setup_name} with cols={cols}, rows={rows}, granularity={granularity}, cores={n_processes}",
                ncols=0,
            )
        )

    # Create the figure and axes to plot on
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(2 * cols, 0.7 + 2.1 * rows),
        sharex=True,
        sharey=True,
    )

    for (data, p2idx, param, value, realized_probs), ax in zip(
        strategy_data, axs.flatten()
    ):
        if idx_map:
            new_data = np.zeros_like(data)
            for fromm, to in idx_map.items():
                new_data[data == fromm] = to
            data = new_data

        make_general_strategy_heatmap(
            results=data,
            probs=realized_probs,
            p2idx=p2idx_override or p2idx,
            title=f"{param}={value:.2f}",
            ax=ax,
            gammas=gammas,
            annot=False,
            ax_labels=False,
            num_ticks=5,
        )

    # Show the full plot at the end
    fig.suptitle(f"{setup_name} Equivalence Class Invariance\n")
    fig.text(
        0.5,
        subtitle_location,
        "Default: " + ", ".join(f"{k}={v}" for k, v in default_params.items()),
        size=8,
        ha="center",
        va="center",
        style="italic",
    )

    # Set the x and y labels
    for ax in axs[-1]:
        ax.set_xlabel(r"Discount factor $\gamma$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Confidence level $p$")

    # Determine the output directory
    output_dir = init_outdir(setup_name)

    plt.tight_layout()
    # Save the figure
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename is None:
        filename = f"{output_dir}/{now}_vizualization.png"

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    if save_metadata:
        # Add metadata to the heatmap info
        print("Saving metadata with length", len(strategy_data))
        pickle_data = {
            "strategy_data": strategy_data,
            "grid_dimensions": (rows, cols),
            "search_parameters": search_parameters,
            "default_params": default_params,
            "probs": probs,
            "gammas": gammas,
        }

        # Save the heatmap info to file
        with open(f"{output_dir}/{now}_metadata.pkl", "wb") as f:
            pickle.dump(pickle_data, f)  # type: ignore

    # Show the plot
    if show_plot:
        plt.show()
