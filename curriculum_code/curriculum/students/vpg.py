from typing import Union, Callable

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import get_schedule_fn
from torch.optim import Adam
from stable_baselines3.common import logger
from torch.nn import functional as th_funcs

from curriculum.student import Student
from utilities.reward_utilities import reward_to_go, discount_cumsum

LOG_INTERVAL = 10


# based on https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg
class VPG(Student):
    def __init__(self,
                 policy: BasePolicy,
                 env: GymEnv,
                 learning_rate: Union[float, Callable[[float], float]],
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 max_grad_norm: float = 10,
                 iters_per_episode: int = 1
                 ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.learning_rate = learning_rate
        self.lr_schedule = get_schedule_fn(self.learning_rate)

        self.policy = policy

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.iters_per_episode = iters_per_episode

        self.batch_obs = []
        self.batch_last_obs = None
        self.batch_actions = []
        self.batch_rewards = []

        self._episode_num = 0
        self.episode_rewards = []  # for measuring avg rewards

    def get_action(self, obs, env, deterministic=False):
        action, state = self.policy.predict(obs, None, None, deterministic)
        return action

    def _record_action(self, obs, action, reward, next_obs, done):
        self.batch_obs.append(np.array(obs).copy())
        self.batch_last_obs = np.array(next_obs).copy()
        self.batch_actions.append(np.array(action).copy())
        self.batch_rewards.append(np.array(reward).copy())

    def _before_episode(self):
        self.batch_obs = []
        self.batch_last_obs = None
        self.batch_actions = []
        self.batch_rewards = []

    def _after_episode(self):
        # Log training info
        self._episode_num += 1
        if self._episode_num % LOG_INTERVAL == 0:
            self.__dump_logs()
        pass

    def _handle_done_signal(self):
        self.episode_rewards.append(sum(self.batch_rewards))
        for i in range(self.iters_per_episode):
            self._handle_epoch_end()

    def _handle_epoch_end(self):
        # TODO: clean up this shit. We need to generate all the stuff on end
        obs_tensor = th.as_tensor(np.vstack(self.batch_obs), dtype=th.float32).to("cuda")
        action_tensor = th.as_tensor(np.vstack(self.batch_actions), dtype=th.float32).to("cuda")

        with th.no_grad():
            _, values, _ = self.policy.forward(obs_tensor)

        # Trick: fill in last value using the network estimation (a minor bootstrap for optimization)
        with th.no_grad():
            last_obs_tensor = th.as_tensor([self.batch_last_obs], dtype=th.float32).to("cuda")
            _, last_value, _ = self.policy.forward(last_obs_tensor)
            next_values = th.cat((values, last_value))

        values, log_prob, entropy = self.policy.evaluate_actions(obs_tensor, action_tensor)

        with th.no_grad():
            value_diff = self.gamma * next_values[1:, :] - values
            value_diff = value_diff.flatten()

        ep_rewards = np.vstack(self.batch_rewards).flatten()
        deltas = ep_rewards + value_diff.cpu().numpy()
        advantages = discount_cumsum(deltas, self.gamma * self.gae_lambda)
        advantages_tensor = th.as_tensor(advantages.copy(), dtype=th.float32).to("cuda")

        if entropy is None:
            # Approximate entropy when no analytical form
            ratio = -th.mean(-log_prob)
        else:
            ratio = -th.mean(entropy)
        policy_loss = -(ratio*advantages_tensor).mean()

        returns = np.array(reward_to_go(ep_rewards, gamma=self.gamma))  # discounted return
        returns_tensor = th.as_tensor(returns.copy(), dtype=th.float32).to("cuda")
        value_loss = th_funcs.mse_loss(returns_tensor, values.flatten())

        total_batch_loss = policy_loss + value_loss
        # Optimization step
        self.policy.optimizer.zero_grad()
        total_batch_loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def __dump_logs(self) -> None:
        """
        Write log.
        """
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        mean_reward = np.mean(self.episode_rewards)
        logger.record("mean_reward", mean_reward, exclude="tensorboard")
        #logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")

        # Pass the number of timesteps for tensorboard
        logger.dump(step=self._episode_num)
        #logger.dump(step=self.num_timesteps)
