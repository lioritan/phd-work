import collections
import functools
from collections import deque

from stable_baselines3.common.type_aliases import GymEnv
import random
import numpy as np

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


# Teacher-Student Curriculum Learning algorithm 4
class LearningRateSamplingTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.buffer_size = teacher_parameters["k"]
        self.task_mapping = []
        self.last_task_values = {}
        self.task_reward_buffer = {}

        total_possible_combinations = []
        for env_param in self.env_wrapper.parameters.values():
            total_possible_combinations.append(len(env_param.values))
        self.total_envs = functools.reduce(lambda x, y: x * y, total_possible_combinations)

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        sampled_task_weights = {}
        for task_ind in self.task_reward_buffer.keys():
            reward = random.choice(self.task_reward_buffer[task_ind])
            sampled_task_weights[task_ind] = reward

        if len(sampled_task_weights) == 0:
            chosen_params = self.pick_random_params()
            new_env = self.env_wrapper.create_env(chosen_params)
            return new_env, chosen_params

        task_rates = list(map(abs, sampled_task_weights.values()))
        best_task_ind = np.argmax(task_rates)

        if abs(sampled_task_weights[best_task_ind]) < 1 and self.total_envs > len(sampled_task_weights):
            while True:  # Generate envs until finding a new one
                chosen_params = self.pick_random_params()
                if chosen_params not in self.task_mapping:
                    new_env = self.env_wrapper.create_env(chosen_params)
                    return new_env, chosen_params

        chosen_params = self.task_mapping[best_task_ind].copy()
        new_env = self.env_wrapper.create_env(chosen_params)
        return new_env, chosen_params

    def pick_random_params(self):
        chosen_params = {}
        for param_name in self.env_wrapper.parameters.keys():
            sampled_value = self.env_wrapper.parameters[param_name].sample()  # TODO: seed
            chosen_params[param_name] = sampled_value
        return chosen_params

    def update_teacher_policy(self):
        latest_task, latest_result = self.history[-1]
        if latest_task not in self.task_mapping:
            self.task_mapping.append(latest_task)
            new_ind = len(self.task_mapping) - 1
            self.task_reward_buffer[new_ind] = deque([latest_result], maxlen=self.buffer_size)
            self.last_task_values[new_ind] = latest_result
        else:
            ind = self.task_mapping.index(latest_task)
            self.task_reward_buffer[ind].append(latest_result - self.last_task_values[ind])
            self.last_task_values[ind] = latest_result
