from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
            self,
            size=16,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            walls=[[2, 1]]
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        # ADDED: Attribute walls, a list containing all 2-dim coordinates with walls
        self.walls = walls

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 3, height - 3)

        # Place a goal square in the top-right corner
        self.put_obj(Goal(), width - 3, 3)

        # Place a goal square in the bottom-left corner
        self.put_obj(Goal(), 4, height - 4)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to a green goal square"

        # ADDED: Attribute Walls, a list containing all coordinates where walls are being placed
        for coord in self.walls:
            assert self.grid.get(*coord) is None
            self.put_obj(Wall(), coord[0], coord[1])

        # ADDED: Random Blocks
        # self.put_obj(Wall(), 2, 1)

        # for _ in range(40):
        #     x = random.randint(0, self.width-1)
        #     y = random.randint(0, self.height-1)
        #     if self.grid.get(*[x,y]) == None and [x, y] != [1,1]:
        #         self.put_obj(Wall(), x, y)


class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)


class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)


class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)



register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
