from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class ConstructedMazeEnv(MiniGridEnv):
    """
    Grid environment with specifically placed walls; sparse reward
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            walls=[]
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        # ADDED: Attribute walls, a list containing all 2-dim coordinates with walls
        self.walls = walls

        self.type = 'Maze'

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

        # for 9x9 ------------ORIGINAL---------------
        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 3, height - 3)
        # self.put_obj(Goal(), width - 3, 2)
        # self.put_obj(Goal(), 2, height - 3)
        #
        # # Place lava squares
        # self.grid.set(1, 4, Lava())
        # self.grid.set(3, 2, Lava())
        # self.grid.set(4, 2, Lava())
        #
        # self.grid.set(2, 4, Lava())
        # self.grid.set(6, 5, Lava())
        # self.grid.set(7, 5, Lava())

        # for 9x9 -------------LAVA-BARS--------------
        # goal squares
        # self.agent_start_pos = (1, 4)
        self.put_obj(Goal(), width - 2, height - 5)
        self.put_obj(Goal(), width - 7, 7)
        self.put_obj(Goal(), 5, height - 8)

        self.grid.set(3, 2, Lava())
        self.grid.set(3, 3, Lava())
        self.grid.set(3, 4, Lava())

        self.grid.set(5, 5, Lava())
        self.grid.set(5, 6, Lava())
        self.grid.set(5, 7, Lava())




        # for 8x8
        # # Place a goal square in the top-right corner
        # self.put_obj(Goal(), width - 2, 3)
        #
        # # Place a goal square in the bottom-left corner
        # self.put_obj(Goal(), 2, height - 2)
        #
        # self.grid.set(1, 3, Lava())
        # self.grid.set(2, 3, Lava())
        #
        # self.grid.set(4, 1, Lava())
        # self.grid.set(5, 1, Lava())
        #
        # self.grid.set(5, 5, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to a green goal square"

        # ADDED: Attribute Walls, a list containing all coordinates where walls are being placed
        for coord in self.walls:
            # assert self.grid.get(*coord) is None
            self.put_obj(Wall(), coord[0], coord[1])


class ConstructedMazeEnv16x16(ConstructedMazeEnv):
    def __init__(self, walls, **kwargs):
        super().__init__(size=16, walls=walls)

# register(
#     id='MiniGrid-Empty-16x16-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv16x16'
# )
