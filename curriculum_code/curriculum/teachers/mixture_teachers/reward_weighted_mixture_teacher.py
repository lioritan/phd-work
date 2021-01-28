from stable_baselines3.common.type_aliases import GymEnv
import numpy as np
from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any

from curriculum.teachers.utils.probability_utils import array_to_probability_dist


class RewardMixtureTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.teachers = teacher_parameters["teachers"]
        self.teacher_weights = np.ones(len(self.teachers)) / len(self.teachers)
        self.chosen_teacher = 0

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        choice_idx = np.random.choice(np.arange(len(self.teachers)), size=1, p=self.teacher_weights)
        self.chosen_teacher = int(choice_idx[0])
        return self.teachers[self.chosen_teacher].generate_task()

    def update_teacher_policy(self):
        self.teachers[self.chosen_teacher].history.history.append(self.history[-1])
        self.teachers[self.chosen_teacher].update_teacher_policy()

        mean_rewards = []
        for teacher in self.teachers:
            rewards = [r for t, r in teacher.history.history]
            if len(rewards) == 0:
                mean_rewards.append(np.mean([r for t, r in self.history.history]))
            else:
                mean_rewards.append(np.mean(rewards))
        self.teacher_weights = array_to_probability_dist(np.array(mean_rewards))
