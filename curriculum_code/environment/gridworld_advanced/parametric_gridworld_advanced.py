
from typing import Any, Dict
import operator
from functools import reduce

import gym
from gym_minigrid.wrappers import ImgObsWrapper

from environment.environment_parameter import ContinuousParameter, CategoricalParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.gridworld_advanced.gridworld_env import GridworldEnv


class GridworldsCustomWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Gridworlds Custom"
        self.parameters = {
            "depth": DiscreteParameter(2, 10),
            "width": DiscreteParameter(1, 10),
            "keys": DiscreteParameter(0, 6),
            "maze_percentage": ContinuousParameter(0, 1),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        # note: stable baselines automatically flattens images for the Mlp policy net, and runs 2Dconv for the Cnn
        # policy net, so image observations are fine. This DOES make the net deeper for the Mlp case.
        return ImgObsWrapper(GridworldEnv(parameter_values["depth"], parameter_values["width"], parameter_values["keys"], parameter_values["maze_percentage"]))
