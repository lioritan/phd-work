from curriculum.teachers.reward_shaping.reward_shaping_wrapper_teacher import RewardShapingTeacher
import numpy as np


class NewStateShaping(RewardShapingTeacher):
    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters, base_teacher)
        self.similiarity_dist = teacher_parameters["state_distance"]
        self.scale = teacher_parameters["scale"]
        self.reward = teacher_parameters.get("new_state_reward", None)
        self.known_states = []
        self.mean_reward = 0

    def shaping_function(self, s, a):
        if self.has_similar_state(s):
            tot_reward = 0
        else:
            if self.reward is not None:
                tot_reward = self.scale * self.reward
            else:
                tot_reward = self.scale * self.mean_reward

        self.known_states.append(s)
        return tot_reward

    def update_shaping_function(self):
        self.mean_reward = np.mean([x[1] for x in self.history.history])
        self.known_states = list(set(self.known_states))

    def has_similar_state(self, state):
        state_array = np.array(state)
        for other_state in self.known_states:
            other_state_array = np.array(other_state)
            if np.linalg.norm(state_array - other_state_array) < self.similiarity_dist:
                return True
        return False
