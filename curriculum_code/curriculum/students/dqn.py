from timeit import time
from typing import Optional, Tuple, Type, Dict, Any, Union, List, Callable

import numpy as np
import torch as th
from stable_baselines3.common import logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import is_vectorized_observation, polyak_update, get_linear_fn, \
    update_learning_rate, get_schedule_fn
from torch.nn import functional as th_funcs

from curriculum.student import Student

# Based on the implementation from https://github.com/DLR-RM/stable-baselines3
from utilities.replay_buffer import ReplayBuffer

LOG_INTERVAL = 10


class DQN(Student):
    def __init__(self,
                 policy: BasePolicy,
                 env: GymEnv,
                 learning_rate: Union[float, Callable[[float], float]],
                 target_update_interval: int = 10000,
                 gradient_steps: int = 1,
                 batch_size: int = 32,
                 learning_starts: int = 100,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 max_grad_norm: float = 10,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 buffer_size: int = 1000000,
                 ):
        pass
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_timesteps = 0
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        self.exploration_rate = exploration_initial_eps
        self._explore_annealing = 0

        self.learning_rate = learning_rate
        self.lr_schedule = get_schedule_fn(self.learning_rate)

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
        )

        self.policy = policy
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

        # For logging
        self._n_updates = 0
        self._episode_num = 0
        self.episode_rewards = []

    def get_action(self, obs, env, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:  # epsilon-exploration
            if is_vectorized_observation(obs, self.observation_space):
                n_batch = obs.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(obs, None, None, deterministic)
        return action

    def _record_action(self, obs, action, reward, next_obs, done):
        self.num_timesteps += 1
        self.episode_reward += reward
        self.replay_buffer.add(obs, next_obs, action, reward, done)
        self.__dqn_optional_updates()

        if self.num_timesteps > self.learning_starts:  # minimal time to put some data in the buffer
            self.__train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

    def _handle_done_signal(self, obs):
        self._episode_num += 1
        self.episode_rewards.append(self.episode_reward)
        # Log training infos
        if self._episode_num % LOG_INTERVAL == 0:
            self.__dump_logs()

    def _after_episode(self):
        mean_reward = np.mean(self.episode_rewards)
        logger.record("mean_reward", mean_reward, exclude="tensorboard")

    def _before_episode(self):
        self.episode_reward = 0

    def __dqn_optional_updates(self) -> None:
        """
        Update the exploration rate and target network if needed.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self._explore_annealing = min(1, self._explore_annealing + 0.01)  # custom exploration annealing
        self.exploration_rate = self.exploration_schedule(self._explore_annealing)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def __train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self.__update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            # TODO: normalize by env mean reward/obs/...

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = th_funcs.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def __update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        logger.record("train/learning_rate", self.lr_schedule(self._explore_annealing))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._explore_annealing))

    def __dump_logs(self) -> None:
        """
        Write log.
        """
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        mean_reward = np.mean(self.episode_rewards)
        logger.record("mean_reward", mean_reward, exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        # if self.use_sde:
        #     logger.record("train/std", (self.actor.get_std()).mean().item())

        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)
