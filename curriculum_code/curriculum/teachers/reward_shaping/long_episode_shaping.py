from curriculum.teachers.reward_shaping.reward_shaping_wrapper_teacher import RewardShapingTeacher
import numpy as np


class LongEpisodeShaping(RewardShapingTeacher):

    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters, base_teacher)
        self.scale = teacher_parameters["scale"]
        self.is_strong_shaping = teacher_parameters.get("is_strong", False)
        self.episode_length = 0

    def shaping_function(self, s, a):
        if self.is_strong_shaping:
            return self.scale * self.episode_length
        else:
            return self.scale * 1

    def shaping_step_update_function(self, s, a, r, s_new, done):
        super(LongEpisodeShaping, self).shaping_step_update_function(s, a, r, s_new, done)
        if self.is_strong_shaping:
            self.episode_length += 1

    def update_shaping_function(self):
        self.episode_length = 0
