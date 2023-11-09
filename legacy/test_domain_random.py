from maze_env import ConstructedMazeEnv
from env_design import domain_randomisation
from gym_minigrid.wrappers import StochasticActionWrapper
from auxiliary.auxiliary import get_expert_trajectory


if __name__ == "__main__":
    size = 9
    base_env = ConstructedMazeEnv(size=size)

    while True:
        walls = domain_randomisation(base_env)  # domain randomisation
        env = ConstructedMazeEnv(size=size, walls=walls)
        env = StochasticActionWrapper(env, 0.9)
        env.reset()
        get_expert_trajectory(env)
        env.render()
        i = input("Press enter to continue...")
        if i == "q":
            break
        env.close()

    env.close()
