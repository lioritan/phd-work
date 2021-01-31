from curriculum.teachers.reward_shaping.reward_shaping_wrapper_teacher import RewardShapingTeacher
import numpy as np
import torch
import torch.nn.functional as F

from curriculum.teachers.utils.neural_net_utils import SimpleNet
from curriculum.teachers.utils.task_space_utils import discount_cumsum


class LearnedQShaping(RewardShapingTeacher):

    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters, base_teacher)
        self.scale = teacher_parameters["scale"]
        self.step_size = teacher_parameters["step_size"]
        obs_space_size = teacher_parameters["obs_shape"]
        action_space_size = teacher_parameters["action_shape"]
        self.net = SimpleNet(teacher_parameters["network_dimensions"], obs_space_size + action_space_size, 1) # Nope!
        self.discount = teacher_parameters["discount"]
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = [[]]

    def shaping_function(self, s, a):
        with torch.no_grad():
            obs_tensor = torch.tensor(np.hstack((s, a)), dtype=torch.float32)
            predicted_q = self.net(obs_tensor)
            return self.scale * predicted_q.item()

    def shaping_step_update_function(self, s, a, r, s_new, done):
        super(LearnedQShaping, self).shaping_step_update_function(s, a, r, s_new, done)
        self.episode_obs.append(s)
        self.episode_actions.append(a)
        self.episode_rewards[-1].append(r)
        if done:
            self.episode_rewards.append([])

    def update_shaping_function(self):
        reward_arrays = []
        for ep_rewards in self.episode_rewards:
            reward_arrays.extend(discount_cumsum(np.array(ep_rewards), self.discount))

        reward_array = np.array(reward_arrays)
        obs = torch.tensor(np.hstack((self.episode_obs, self.episode_actions)), dtype=torch.float32)
        task_reward = torch.tensor([reward_array], dtype=torch.float32).transpose(0, 1)
        predict_loss = torch.nn.MSELoss()(task_reward, self.net(obs)) * self.step_size
        self.net.zero_grad()
        predict_loss.backward()

        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = [[]]
