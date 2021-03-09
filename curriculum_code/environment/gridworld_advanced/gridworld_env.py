import random

from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Key, Door, Lava, Wall, COLOR_NAMES, Floor
from mazelib import Maze
from mazelib.generate.AldousBroder import AldousBroder


ROOM_SIZE = 5  # assumed to be odd


class GridworldEnv(MiniGridEnv):
    """
    Environment of multiple 5x5 rooms
    """

    def __init__(self, depth, width, keys, maze_percentage):
        self.rooms_depth = depth
        self.rooms_width = width
        self.num_keys = keys
        self.maze_percentage = maze_percentage

        super().__init__(
            width=(ROOM_SIZE + 2) * depth,
            height=(ROOM_SIZE + 2) * width,
            max_steps=10 * (ROOM_SIZE ** 2) * depth * width,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        unused_key_colors = []
        maze_rooms = round(self.maze_percentage * (self.rooms_depth - 2) * self.rooms_width)

        # add walls to the rooms
        for room_col in range(self.rooms_depth):
            for room_row in range(self.rooms_width):
                col_idx = room_col * (ROOM_SIZE + 2)
                row_idx = room_row * (ROOM_SIZE + 2)

                # add horizontal walls to the room
                for c_shift in range(ROOM_SIZE + 2):
                    for r_shift in [0, ROOM_SIZE + 1]:
                        self.put_obj(Wall(), col_idx + c_shift, row_idx + r_shift)

                # add vertical walls to the room
                for r_shift in range(ROOM_SIZE + 2):
                    for c_shift in [0, ROOM_SIZE + 1]:
                        self.put_obj(Wall(), col_idx + c_shift, row_idx + r_shift)

                # generate room
                if room_col == 0 and room_row == 0:  # empty start room
                    self.agent_pos = (self._rand_int(1, ROOM_SIZE + 1), self._rand_int(1, ROOM_SIZE + 1))
                    self.agent_dir = self._rand_int(0, 4)
                elif room_col == self.rooms_depth - 1 and room_row == 0:  # empty goal room
                    self.put_obj(Goal(), col_idx + self._rand_int(1, ROOM_SIZE + 1), self._rand_int(1, ROOM_SIZE + 1))
                else:
                    if maze_rooms > 0:
                        if self.coin_flip():
                            maze_rooms -= 1
                            m = Maze()
                            m.generator = AldousBroder((ROOM_SIZE + 1) / 2, (ROOM_SIZE + 1) / 2)  # includes outer wall
                            m.generate()

                            barrier_type = random.choice([Wall, Lava])

                            for c_shift in range(1, ROOM_SIZE + 1):
                                for r_shift in range(1, ROOM_SIZE + 1):
                                    if m.grid[c_shift, r_shift] == 1:
                                        self.put_obj(barrier_type(), col_idx + c_shift, row_idx + r_shift)

                    if len(unused_key_colors) < len(COLOR_NAMES) and len(unused_key_colors) < self.num_keys:
                        if self.coin_flip():
                            key_color = random.choice(list(set(COLOR_NAMES).difference(unused_key_colors)))
                            unused_key_colors.append(key_color)
                            key_offset_c, key_offset_r = self.find_free_spot(col_idx, row_idx)
                            self.put_obj(Key(color=key_color), col_idx + key_offset_c, row_idx + key_offset_r)

        for room_col in range(self.rooms_depth):
            for room_row in range(self.rooms_width):
                col_idx = room_col * (ROOM_SIZE + 2)
                row_idx = room_row * (ROOM_SIZE + 2)
                # connect top - bottom
                if self.rooms_width > 1 and room_row < self.rooms_width - 1:
                    for c_shift in range(1, ROOM_SIZE + 1):
                        if self.grid.get(col_idx + c_shift, row_idx + ROOM_SIZE) is None:
                            add_door = self.coin_flip() and room_row > 0
                            if add_door and len(unused_key_colors) > 0:
                                obj_to_place = Door(color=unused_key_colors[0])
                                unused_key_colors.pop(0)
                            else:
                                obj_to_place = None
                            self.grid.set(col_idx + c_shift, row_idx + ROOM_SIZE + 1, obj_to_place)
                            self.grid.set(col_idx + c_shift, row_idx + ROOM_SIZE + 2, None)
                            break
                    pass

                # connect left - right
                if room_col < self.rooms_depth - 1:
                    for r_shift in range(1, ROOM_SIZE + 1):
                        if self.grid.get(col_idx + ROOM_SIZE, row_idx + r_shift) is None:
                            add_door = self.coin_flip() and room_col > 0
                            if add_door and len(unused_key_colors) > 0:
                                obj_to_place = Door(color=unused_key_colors[0])
                                unused_key_colors.pop(0)
                            else:
                                obj_to_place = None
                            self.grid.set(col_idx + ROOM_SIZE + 1, row_idx + r_shift, obj_to_place)
                            self.grid.set(col_idx + ROOM_SIZE + 2, row_idx + r_shift, None)
                            break
                    pass

        self.mission = "get to the goal"

    def find_free_spot(self, room_i, room_j):
        free_spots = []
        for i in range(1, ROOM_SIZE + 1):
            for j in range(1, ROOM_SIZE + 1):
                if self.grid.get(room_i + i, room_j + j) is None:
                    free_spots.append((i, j))
        return random.choice(free_spots)

    def coin_flip(self):
        return self._rand_int(0, 2) > 0
