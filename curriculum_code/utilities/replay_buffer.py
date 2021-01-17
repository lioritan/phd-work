from stable_baselines3.common.preprocessing import get_obs_shape, get_action_dim
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ReplayBuffer(object):
    def __init__(self, buffer_size, observation_space, action_space):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, ) + self.obs_shape, dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, ) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            self.dones[batch_inds],
            self.rewards[batch_inds],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(th.device("cuda"))
        return th.as_tensor(array).to(th.device("cuda"))

    def add(self, last_obs, new_obs, action, reward, done):
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(last_obs).copy()
        self.next_observations[self.pos] = np.array(new_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
