import wandb
from stable_baselines3.common.type_aliases import GymEnv
import random
import numpy as np

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


# Teacher-Student Curriculum Learning algorithm 1
class LearningRateCustomTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.alpha = teacher_parameters["alpha"]
        self.epsilon = teacher_parameters["epsilon"]
        self.possible_tasks = teacher_parameters["task_set"]
        self.task_mapping = []
        self.task_rate = {}
        self.last_task_values = {}

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        if len(self.task_mapping) == 0 or random.random() < self.epsilon:  # pick randomly
            chosen_params = random.choice(self.possible_tasks)
        else:
            task_rates = list(map(abs, self.task_rate.values()))
            best_rate_ind = np.argmax(task_rates)
            chosen_params = self.task_mapping[best_rate_ind].copy()

        new_env = self.env_wrapper.create_env(chosen_params)
        return new_env, chosen_params

    def update_teacher_policy(self):
        latest_task, latest_result = self.history[-1]

        last_one = self.last_task_values[self.task_mapping.index(latest_task)] if latest_task in self.task_mapping else 0
        wandb.log({"task_num": len(self.history.history),
                   "actual_ALP": abs(latest_result - last_one),
                   "actual_LP": latest_result - last_one})

        if latest_task not in self.task_mapping:
            self.task_mapping.append(latest_task)
            new_ind = len(self.task_mapping) - 1
            self.task_rate[new_ind] = self.alpha * latest_result
            self.last_task_values[new_ind] = latest_result
        else:
            ind = self.task_mapping.index(latest_task)
            learning_rate = latest_result - self.last_task_values[ind]
            self.task_rate[ind] = self.alpha * learning_rate + (1 - self.alpha) * self.task_rate[ind]
            self.last_task_values[ind] = latest_result


