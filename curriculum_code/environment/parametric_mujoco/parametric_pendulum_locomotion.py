from typing import Any, Dict

import torch
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.envs.classic_control.pendulum import angle_normalize

from environment.environment_parameter import ContinuousParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_mujoco.locomotion_base import StateActionReward


class PendRewardModel(StateActionReward):
    def __init__(self, angle):
        self.angle = angle
        super(PendRewardModel, self).__init__(dim_action=(1,), ctrl_cost_weight=0.001, sparse=False, action_scale=1)

    def state_reward(self, state, next_state=None):
        return -(self.angle - angle_normalize(state[..., 0])) ** 2 - 0.1 * state[..., 1] ** 2


class AngledPendulumEnv(PendulumEnv):
    def __init__(self, angle=0.0):
        super(AngledPendulumEnv, self).__init__()
        self.angle = angle_normalize(angle)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        angle_cost = (self.angle - angle_normalize(th)) ** 2
        costs = angle_cost + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi/8, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def reward_model(self):
        """Get reward model."""
        return PendRewardModel(self.angle)


class MBPendulumAngleContinuousWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "MBPendulum-v2"
        self.parameters = {
            "goal_angle": ContinuousParameter(0.0, 2 * np.pi),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        environment = AngledPendulumEnv(angle=parameter_values["goal_angle"])
        environment.reset()
        return environment
