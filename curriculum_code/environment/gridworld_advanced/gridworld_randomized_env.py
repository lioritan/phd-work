import random
from typing import Any, Dict, Set

from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Key, Door, Lava, Wall, Ball, Floor, OBJECT_TO_IDX, \
    IDX_TO_OBJECT, COLOR_NAMES


# can pick up key, opens door of same color
# ball = moving obstacle, reduces reward
# lava kills, wall blocks
# colored floor can be used for custom objectives

def random_color():
    return random.choice(COLOR_NAMES)


def missing_or_random_color(required_set: Set[str], given_set: Set[str]):
    missing_colors = required_set.difference(given_set)
    if len(missing_colors) > 0:
        return random.choice(list(missing_colors))
    else:
        return random_color()


class GridworldRandomizedEnv(MiniGridEnv):
    """
    Environment where the placement of all elements is controllable
    """

    def __init__(self, positions, goal_pos: int, start_pos: int):
        self.positions = positions
        self.goal = goal_pos
        self.start = start_pos
        self.mission = "get to the goal"

        super().__init__(
            grid_size=(32+2),
            max_steps=10 * 32 * 32
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # TODO: make walls
        for col in range(32+2):
            for row in [0, 32+1]:
                self.put_obj(Wall(), col, row)

        for row in range(32+2):
            for col in [0, 32+1]:
                self.put_obj(Wall(), col, row)

        row = 1
        col = 1
        door_colors = set()
        key_colors = set()
        for i, cell in enumerate(self.positions):
            object_type = IDX_TO_OBJECT[cell]
            if i == self.start:
                self.agent_pos = (col, row)
                self.agent_dir = self._rand_int(0, 4)
            elif i == self.goal:
                self.put_obj(Goal(), col, row)
            elif object_type == 'wall':
                self.put_obj(Wall(), col, row)
            elif object_type == 'key':
                key_color = missing_or_random_color(door_colors, key_colors)
                key_colors.add(key_color)
                self.put_obj(Key(color=key_color), col, row)
            elif object_type == 'door':
                door_color = random_color()
                door_colors.add(door_color)
                self.put_obj(Door(color=door_color, is_locked=True), col, row)
            elif object_type == 'lava':
                self.put_obj(Lava(), col, row)
            else:  # empty tile
                pass
            col = (col + 1) % (width - 1)
            if col == 0:
                col = 1
                row += 1
