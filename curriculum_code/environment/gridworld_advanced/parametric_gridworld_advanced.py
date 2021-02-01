
from typing import Any, Dict

from environment.environment_parameter import ContinuousParameter, CategoricalParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper


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
        return None
