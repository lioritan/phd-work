
from typing import Any, Dict

from environment.environment_parameter import ContinuousParameter, CategoricalParameter
from environment.environment_wrapper import EnvironmentWrapper


class GridworldsCustomWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Gridworlds Custom"
        self.parameters = {
            "depth": CategoricalParameter([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "width": CategoricalParameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "keys": CategoricalParameter(list(range(7))),
            "maze_percentage": ContinuousParameter(0, 1),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        return None
