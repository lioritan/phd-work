from typing import Any, Dict

import gym

from environment.environment_parameter import CategoricalParameter, ContinuousParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_walker_env.parametric_continuous_flat_parkour import ParametricContinuousWalker

ROOF_SIZE = "climbing_surface_size"
GAP_POS = "gap_pos"
GAP_SIZE = "gap_size"
OBS_DIST = "obstacle_spacing"


# Walker2d variations taken from https://github.com/flowersteam/meta-acl/tree/main/teachDRL/gym_flowers/envs
class WalkerWrapper(EnvironmentWrapper):
    def __init__(self, walker_type, walker_params):
        super().__init__()
        self.name = f"ParamWalker-{walker_type}"
        self.body_type = walker_type
        self.parameters = {
            ROOF_SIZE: ContinuousParameter(0.0, 1.0),  # roof obstacle size
            GAP_POS: ContinuousParameter(2.5, 10.0),
            GAP_SIZE: ContinuousParameter(0.0, 20.0),
            OBS_DIST: ContinuousParameter(0.01, 10.0),
        }
        self.parameters.update(walker_params)

    def create_env(self, parameter_values: Dict[str, Any]):
        body_params = {k: v for (k, v) in parameter_values.items() if
                       k not in (ROOF_SIZE, GAP_POS, GAP_SIZE, OBS_DIST)}
        env = ParametricContinuousWalker(0, self.body_type, **body_params)
        env.set_environment(gap_pos=parameter_values[GAP_POS], obstacle_spacing=parameter_values[OBS_DIST],
                            gap_size=parameter_values[GAP_SIZE], climbing_surface_size=parameter_values[ROOF_SIZE])
        env.reset()
        return env


def get_classic_walker():
    return WalkerWrapper(walker_type="classic_bipedal", walker_params={
        "motors_torque": ContinuousParameter(1, 160)
    })


def get_small_walker():
    return WalkerWrapper(walker_type="small_bipedal", walker_params={
        "motors_torque": ContinuousParameter(1, 160)
    })


def get_human_walker():
    return WalkerWrapper(walker_type="human", walker_params={
        "motors_torque": ContinuousParameter(1, 200)
    })


def get_monkey_walker():
    return WalkerWrapper(walker_type="profile_chimpanzee", walker_params={
        "motors_torque": ContinuousParameter(1, 200)
    })


def get_quadro_walker():
    return WalkerWrapper(walker_type="big_quadru", walker_params={
        "motors_torque": ContinuousParameter(1, 500)
    })


def get_spider_walker():
    return WalkerWrapper(walker_type="spider", walker_params={
        "motors_torque": ContinuousParameter(1, 200),
        "nb_pairs_of_legs": CategoricalParameter(list(range(2, 9)))
    })


def get_millipede_walker():
    return WalkerWrapper(walker_type="millipede", walker_params={
        "motors_torque": ContinuousParameter(1, 500),
        "nb_of_bodies": CategoricalParameter(list(range(2, 9)))
    })


def get_wheel_walker():
    return WalkerWrapper(walker_type="wheel", walker_params={
        "motors_torque": ContinuousParameter(100, 1000),
        "body_scale": ContinuousParameter(0.1, 1.5)
    })
