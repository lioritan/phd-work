
from typing import Any, Dict

from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.wrappers import ImgObsWrapper

from environment.environment_parameter import ContinuousParameter, CategoricalParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.gridworld_advanced.gridworld_randomized_env import GridworldRandomizedEnv


class GridworldsRandomizedWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Gridworlds randomized"
        self.parameters = {
            "start_pos": DiscreteParameter(0, 32*32),
            "goal_pos": DiscreteParameter(0, 32*32),
        }
        for i in range(32*32):
            self.parameters[f"pos {i}"] = DiscreteParameter(0, len(IDX_TO_OBJECT)-1)

    def create_env(self, parameter_values: Dict[str, Any]):
        start = parameter_values["start_pos"]
        goal = parameter_values["goal_pos"]
        positions = [parameter_values[f"pos {i}"] for i in range(32*32)]
        return ImgObsWrapper(GridworldRandomizedEnv(positions, goal, start))
