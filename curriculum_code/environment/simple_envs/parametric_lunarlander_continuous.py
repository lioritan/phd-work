from typing import Any, Dict

import gym
import gym.envs.box2d.lunar_lander as lander

from environment.environment_parameter import ContinuousParameter
from environment.environment_wrapper import EnvironmentWrapper


class LunarLanderContinuousWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "LunarLanderContinuous-v2"
        self.parameters = {
            "leg_height": ContinuousParameter(5, 50),
            "leg_width": ContinuousParameter(2, 10),
            "main_engine_power": ContinuousParameter(5.0, 100.0),
            "side_engine_power": ContinuousParameter(0.0, 10.0),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        base_env = gym.make("LunarLanderContinuous-v2")
        lander.MAIN_ENGINE_POWER = parameter_values["main_engine_power"]
        lander.SIDE_ENGINE_POWER = parameter_values["side_engine_power"]
        lander.LEG_H = parameter_values["leg_height"]
        lander.LEG_W = parameter_values["leg_width"]
        lander.INITIAL_RANDOM = 1500

        base_env.reset()
        return base_env
