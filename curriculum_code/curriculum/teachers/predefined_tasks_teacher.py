from stable_baselines3.common.type_aliases import GymEnv

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


#  teacher with a set of pre-defined tasks to cycle through (includes const task)
class PredefinedTasksTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)

        self.tasks_to_cycle = teacher_parameters["tasks"]
        self.idx = 0

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        new_task_params = self.tasks_to_cycle[self.idx]
        self.idx = (self.idx + 1) % len(self.tasks_to_cycle)
        new_env = self.env_wrapper.create_env(new_task_params)
        return new_env, new_task_params

    def update_teacher_policy(self):
        pass
