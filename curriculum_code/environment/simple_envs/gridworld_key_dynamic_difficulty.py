import gym_minigrid

from typing import Any, Dict

import gym
from gym_minigrid.wrappers import ImgObsWrapper

from environment.environment_parameter import ContinuousParameter, CategoricalParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper


class GridworldKeyWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Gridworld key corridor"  # a classic curriculum learning env
        self.parameters = {
            "difficulty": DiscreteParameter(1, 6),
        }
        self.envs = ["MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS3R2-v0", "MiniGrid-KeyCorridorS3R3-v0",
                     "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS5R3-v0", "MiniGrid-KeyCorridorS6R3-v0"]

    def create_env(self, parameter_values: Dict[str, Any]):
        difficulty = parameter_values["difficulty"] - 1
        base_env = gym.make(self.envs[difficulty])

        base_env.reset()
        return ImgObsWrapper(base_env)
