from stable_baselines3.common.type_aliases import GymEnv
import random
import numpy as np

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


class LearningRateSamplingTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.alpha = teacher_parameters["alpha"]
        self.epsilon = teacher_parameters["epsilon"]
        self.task_mapping = []
        self.task_rate = {}
        self.last_task_values = {}

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        pass

    def update_teacher_policy(self):
        pass
