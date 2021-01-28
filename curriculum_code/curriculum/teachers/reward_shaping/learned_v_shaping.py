from curriculum.teachers.reward_shaping.reward_shaping_wrapper_teacher import RewardShapingTeacher
import numpy as np
import torch

from curriculum.teachers.utils.neural_net_utils import SimpleNet


class LearnedVShaping(RewardShapingTeacher):

    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters, base_teacher)
        self.scale = teacher_parameters["scale"]
        self.step_size = teacher_parameters["step_size"]
        obs_space_size = teacher_parameters["obs_shape"]
        self.net = SimpleNet(teacher_parameters["network_dimensions"], obs_space_size, 1)
        self.discount = teacher_parameters["discount"]
        self.episode_obs = []
        self.episode_rewards = []

    def shaping_function(self, s, a):
        self.episode_obs.append(s)
        with torch.no_grad():
            obs_tensor = torch.tensor(s, dtype=torch.float32)
            predicted_v = self.net(obs_tensor)
            return self.scale * predicted_v.item()

    def shaping_step_update_function(self, s, a, r, s_new, done):
        self.episode_obs.append(s)
        self.episode_rewards.append(r + self.discount * self.episode_rewards[-1])
        if done:
            pass  # TODO: ?

    def update_shaping_function(self):
        obs = torch.tensor(self.episode_obs, dtype=torch.float32)
        task_reward = torch.tensor(self.episode_rewards, dtype=torch.float32)
        predict_loss = torch.nn.MSELoss()(task_reward, self.net(obs))
        self.net.zero_grad()
        predict_loss.backward()  # TODO: step size?

        self.episode_obs = []
        self.episode_rewards = []
