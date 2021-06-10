from typing import Any, Dict

import torch
import numpy as np
from gym import spaces
from gym.envs.classic_control import PendulumEnv
from gym.envs.classic_control.pendulum import angle_normalize

from environment.environment_parameter import ContinuousParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_mujoco.locomotion_base import StateActionReward


class PendRewardModel(StateActionReward):
    def __init__(self, angle):
        self.angle = angle
        self.cos_angle = np.cos(angle)
        self.sin_angle = np.sin(angle)
        super(PendRewardModel, self).__init__(dim_action=(1,), ctrl_cost_weight=0.001, sparse=False, action_scale=1)

    def state_reward(self, state, next_state=None):
        angle_cost = -0.5 * (self.cos_angle - state[..., 0]) ** 2 - 0.5 * (self.sin_angle - state[..., 1]) ** 2
        return angle_cost - 0.1 * state[..., 2] ** 2


class AngledPendulumEnv(PendulumEnv):
    def __init__(self, angle=0.0, randomize_start=False, goal_in_state=False):
        super(AngledPendulumEnv, self).__init__()
        self.angle = angle
        self.m = 0.1
        self.goal_in_state = goal_in_state
        self.radomize_start = randomize_start

        if self.goal_in_state:
            high = np.array([1., 1., self.max_speed, 1., 1.], dtype=np.float32)
        else:
            high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        angle_cost = angle_normalize(self.angle - th) ** 2
        costs = angle_cost + .1 * thdot ** 2 #+ .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reward_model(self):
        """Get reward model."""
        return PendRewardModel(self.angle)

    def reset(self):
        if self.radomize_start:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        if self.goal_in_state:
            return np.array([np.cos(theta), np.sin(theta), thetadot, np.cos(self.angle), np.sin(self.angle)])
        else:
            return np.array([np.cos(theta), np.sin(theta), thetadot])


class MBPendulumAngleContinuousWrapper(EnvironmentWrapper):
    def __init__(self, randomize_start=False, goal_in_state=False):
        super().__init__()
        self.name = "MBPendulum-v2"
        self.parameters = {
            "goal_angle": ContinuousParameter(0.0, 2 * np.pi),
        }

        self.randomize_start = randomize_start
        self.goal_in_state = goal_in_state

    def create_env(self, parameter_values: Dict[str, Any]):
        environment = AngledPendulumEnv(angle=parameter_values["goal_angle"],
                                        randomize_start=self.randomize_start,
                                        goal_in_state=self.goal_in_state)
        environment.reset()
        return environment
