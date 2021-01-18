from typing import Any, Dict

import gym

from environment.environment_parameter import CategoricalParameter
from environment.environment_wrapper import EnvironmentWrapper


class CartpoleWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "CartPole-v1"
        self.parameters = {
            "pole_length": CategoricalParameter([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            "cart_mass": CategoricalParameter([0.8, 0.9, 1.0, 1.1, 1.2]),
            "pole_mass": CategoricalParameter([0.05, 0.1, 0.2]),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        base_env = gym.make("CartPole-v1")
        base_env.masscart = parameter_values["cart_mass"]
        base_env.masspole = parameter_values["pole_mass"]
        base_env.total_mass = (base_env.masspole + base_env.masscart)
        base_env.length = parameter_values["pole_length"]
        base_env.polemass_length = (base_env.masspole * base_env.length)
        base_env.reset()
        return base_env
