from typing import Any, Dict

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

from environment.environment_parameter import ContinuousParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_mujoco.locomotion_base import LocomotionEnv, StateActionReward


class HalfCheetahRewardModel(StateActionReward):
    def __init__(self, dim_action, expected_speed, ctrl_cost_weight=0.1, action_scale=1.0, forward_reward_weight=1.0,
                 healthy_reward=0.0):
        self.expected_speed = expected_speed
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        super(HalfCheetahRewardModel, self).__init__(dim_action, ctrl_cost_weight, False, action_scale)

    def state_reward(self, state, next_state=None):
        # state[0] is velocity, state[1:9] are the rest of the position (no x pos), state[9:18] are velocities
        speed = state[..., 0]  #
        print(speed[0], state[0, 9])
        return -self.forward_reward_weight * torch.abs(speed - self.expected_speed) + self.healthy_reward


class MBHalfCheetahEnv(LocomotionEnv, HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, ctrl_cost_weight=0.1, expected_speed=0.0):
        self.base_mujoco_name = "HalfCheetah-v3"
        self.expected_speed = expected_speed
        LocomotionEnv.__init__(
            self,
            dim_pos=1,
            reward_model=HalfCheetahRewardModel((6,), expected_speed, ctrl_cost_weight)
        )
        HalfCheetahEnv.__init__(
            self, ctrl_cost_weight=ctrl_cost_weight, forward_reward_weight=1.0
        )
        self.dim_state = (18,)
        self.dim_action = (6,)
        self.num_states = None
        self.num_actions = None
        self.goal = None

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        # Note: Changed to be difference from expected speed
        forward_cost = self._forward_reward_weight * np.abs(x_velocity - self.expected_speed)

        observation = self._get_obs()
        reward = -forward_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'expected_velocity': self.expected_speed,

            'reward_run': -forward_cost,
            'reward_ctrl': -ctrl_cost
        }
        return observation, reward, done, info

    def _get_obs(self):
        return LocomotionEnv._get_obs(self)


class HalfCheetahWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "MBHalfCheetah-v0"
        self.parameters = {
            "expected_speed": ContinuousParameter(0.0, 2.0),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        environment = MBHalfCheetahEnv(expected_speed=parameter_values["expected_speed"])
        return environment
