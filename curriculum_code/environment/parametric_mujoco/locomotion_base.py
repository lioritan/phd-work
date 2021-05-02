"""Python Script Template."""
import gym
import numpy as np
import torch

try:
    from gym.envs.mujoco.humanoid_v3 import HumanoidEnv, mass_center
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    HumanoidEnv, mass_center = None, None


class LargeStateTermination(torch.nn.Module):
    """Hopper Termination Function."""

    def __init__(
            self,
            z_dim=None,
            healthy_state_range=(-100, 100),
            healthy_z_range=(-np.inf, np.inf),
            healthy_angle_range=(-np.inf, np.inf),
    ):
        super().__init__()
        self._info = {}
        self.z_dim = z_dim
        self.healthy_state_range = healthy_state_range
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range

    def copy(self):
        """Get copy of termination model."""
        return LargeStateTermination(
            z_dim=self.z_dim,
            healthy_state_range=self.healthy_state_range,
            healthy_z_range=self.healthy_state_range,
            healthy_angle_range=self.healthy_angle_range,
        )

    @staticmethod
    def in_range(state, min_max_range):
        """Check if state is in healthy range."""
        min_state, max_state = min_max_range
        return (min_state < state) * (state < max_state)

    def is_healthy(self, state):
        """Check if state is healthy."""
        if self.z_dim is None:
            return self.in_range(state, min_max_range=self.healthy_state_range).all(-1)
        z = state[..., self.z_dim]
        angle = state[..., self.z_dim + 1]
        other = state[..., self.z_dim + 1:]

        return (
                self.in_range(z, min_max_range=self.healthy_z_range)
                * self.in_range(angle, min_max_range=self.healthy_angle_range)
                * self.in_range(other, min_max_range=self.healthy_state_range).all(-1)
        )

    def forward(self, state, action, next_state=None):
        """Return termination model logits."""
        if not isinstance(state, torch.Tensor):
            return ~self.is_healthy(state)
        done = ~self.is_healthy(state)
        return (
            torch.zeros(*done.shape, 2)
                .scatter_(dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf"))
                .squeeze(-1)
        )


class StateActionReward(torch.nn.Module):
    r"""Base class for state-action reward functions.

    The reward is computed as:
        ..math:: r = r_{state} + \alpha r_{action},

    where r_{state} is an environment dependent reward function (to be implemented),
    r_{action} is the action cost, and \alpha is set by `ctrl_cost_weight'.

    the action reward is given by:
       ..math:: r_{action} = - \sum_{i=1}^{d} a_i^2, in non-sparse environments.
       ..math:: r_{action} =  e^{-\sum_{i=1}^{d} (a_i/scale_i)^2} - 1 in sparse envs.

    Parameters
    ----------
    ctrl_cost_weight: float, optional (default = 0.1)
        action cost ratio that weights the action to state ratio.
    sparse: bool, optional (default = False).
        flag that indicates whether the reward is sparse or global.
    goal: Tensor, optional (default = None).
        Goal position, optional.
    action_scale: float, optional (default = 1.0).
        scale of action for sparse environments.
    """

    def __init__(self, dim_action, ctrl_cost_weight=0.1, sparse=False, action_scale=1.0):
        super().__init__()
        self.dim_action = dim_action

        self.action_scale = action_scale
        self.ctrl_cost_weight = ctrl_cost_weight
        self.sparse = sparse

    def forward(self, state, action, next_state=None):
        """Get reward distribution for state, action, next_state."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(state, next_state)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl

        try:
            self._info.update(
                reward_state=reward_state.sum().item(),
                reward_ctrl=reward_ctrl.sum().item(),
            )
            reward = reward.type(torch.get_default_dtype()).unsqueeze(-1)
        except AttributeError:
            pass
        return reward, torch.zeros_like(reward).unsqueeze(-1)

    @staticmethod
    def action_non_sparse_reward(action):
        """Get action non-sparse rewards."""
        return -(action ** 2).sum(-1)

    def action_reward(self, action):
        """Get reward that corresponds to action."""
        action = action[..., : self.dim_action[0]]  # get only true dimensions.
        return self.action_non_sparse_reward(action / self.action_scale)

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        raise NotImplementedError


class LocomotionEnv(object):
    """Base Locomotion environment. Is a hack to avoid repeated code."""

    def __init__(
            self,
            dim_pos,
            reward_model,
    ):
        self.dim_pos = dim_pos
        self.prev_pos = np.zeros(dim_pos)
        self._reward_model = reward_model
        self.reward_range = ()
        self._termination_model = LargeStateTermination()

    def step(self, action):
        """See gym.Env.step()."""
        raise NotImplementedError()

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        if isinstance(self, HumanoidEnv):
            x_position = mass_center(self.model, self.sim)
        else:
            x_position = position[: self.dim_pos]
        forward_vel = (x_position - self.prev_pos) / self.dt
        return np.concatenate((forward_vel, position[self.dim_pos:], velocity)).ravel()

    def reset_model(self):
        """Reset model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        qpos[: self.dim_pos] = np.zeros(self.dim_pos).copy()
        self.prev_pos = -self.dt * qvel[: self.dim_pos].copy()
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def termination_model(self):
        """Get default termination model."""
        return LargeStateTermination()
