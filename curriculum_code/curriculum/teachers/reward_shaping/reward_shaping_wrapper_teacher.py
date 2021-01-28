from abc import ABC, abstractmethod

import gym
from stable_baselines3.common.type_aliases import GymEnv

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


class ShapedRewardWrapperEnv(gym.Env):
    def __init__(self, base_env: GymEnv, shaping_function, shaping_update_function):
        super().__init__()
        self.base_env = base_env
        self.shaping = shaping_function
        self.shaping_update = shaping_update_function
        self.last_state = None

        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

    def step(self, action):
        next_state, base_reward, done, info = self.base_env.step(action)
        new_reward = self.shaping(self.last_state, action) + base_reward
        self.shaping_update(self.last_state, action, base_reward, new_reward, done)
        self.last_state = next_state
        return next_state, new_reward, done, info

    def reset(self):
        obs = self.base_env.reset()
        self.last_state = obs
        return obs

    def render(self, mode='human'):
        return self.base_env.render(mode)


class RewardShapingTeacher(Teacher, ABC):
    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters)
        self.wrapped_teacher = base_teacher

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        env, params = self.wrapped_teacher.generate_task()

        shaped_env = ShapedRewardWrapperEnv(env, self.shaping_function, self.shaping_step_update_function)
        return shaped_env, params

    def update_teacher_policy(self):
        self.wrapped_teacher.history.history.append(self.history[-1])
        self.update_shaping_function()

    @abstractmethod
    def shaping_function(self, s, a) -> float:
        pass

    @abstractmethod
    def shaping_step_update_function(self, s, a, r, s_new, done):
        pass

    @abstractmethod
    def update_shaping_function(self):
        pass
