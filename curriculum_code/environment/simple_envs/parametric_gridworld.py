import gym_minigrid

from typing import Any, Dict

import gym

from environment.environment_parameter import ContinuousParameter, CategoricalParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper


class GridworldsWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Gridworlds"
        self.parameters = {
            "room_size": CategoricalParameter([3, 4, 6, 14]),
            "has_key": DiscreteParameter(0, 1),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        room_size = parameter_values["room_size"] + 2  # buffer walls don't count
        if parameter_values["has_key"] == 1:
            base_env = gym.make(f"MiniGrid-DoorKey-{room_size}x{room_size}-v0")
        else:
            base_env = gym.make(f"MiniGrid-Empty-{room_size}x{room_size}-v0")

        base_env.reset()
        return base_env
