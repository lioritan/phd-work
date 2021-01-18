from stable_baselines3.common.type_aliases import GymEnv

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


class RandomTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        chosen_params = {}
        for param_name in self.env_wrapper.parameters.keys():
            sampled_value = self.env_wrapper.parameters[param_name].sample()  # TODO: seed
            chosen_params[param_name] = sampled_value
        new_env = self.env_wrapper.create_env(chosen_params)
        return new_env, chosen_params

    def update_teacher_policy(self):
        pass
