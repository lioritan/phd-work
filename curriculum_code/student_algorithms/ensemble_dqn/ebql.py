from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from typing import Type, Union, Callable, Optional, Dict, Any, Tuple

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common import logger

from student_algorithms.ensemble_dqn.ebql_policies import EBQLEnsemblePolicy


class EBQL(OffPolicyAlgorithm):
    def __init__(self,
                 policy: Type[BasePolicy],
                 env: Union[GymEnv, str],
                 ensemble_size: int,
                 learning_rate: Union[float, Callable] = 1e-4,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 n_episodes_rollout: int = -1,
                 action_noise: Optional[ActionNoise] = None,
                 optimize_memory_usage: bool = False,
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = "auto",
                 support_multi_env: bool = False,
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 sde_support: bool = True,
                 remove_time_limit_termination: bool = False,
                 ):
        super().__init__(policy=policy,
                         env=env,
                         policy_base=EBQLEnsemblePolicy,
                         learning_rate=learning_rate,
                         buffer_size=buffer_size,
                         learning_starts=learning_starts,
                         batch_size=batch_size,
                         tau=tau,
                         gamma=gamma,
                         train_freq=1,
                         gradient_steps=1,
                         n_episodes_rollout=n_episodes_rollout,
                         action_noise=action_noise,
                         optimize_memory_usage=optimize_memory_usage,
                         policy_kwargs=policy_kwargs,
                         tensorboard_log=tensorboard_log,
                         verbose=verbose,
                         device=device,
                         support_multi_env=support_multi_env,
                         create_eval_env=create_eval_env,
                         monitor_wrapper=monitor_wrapper,
                         seed=seed,
                         use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq,
                         use_sde_at_warmup=use_sde_at_warmup,
                         sde_support=False, # TODO: consider this
                         remove_time_limit_termination=remove_time_limit_termination
                         )

        self.ensemble_size = ensemble_size
        if policy_kwargs is None:
            self.policy_kwargs = {}
        self.policy_kwargs["ensemble_size"] = ensemble_size

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self._setup_model()

    def _setup_model(self) -> None:
        super(EBQL, self)._setup_model()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Update learning rate according to schedule

        losses = []
        for gradient_step in range(gradient_steps):  # always 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # pick network
            update_net_index = np.random.choice(np.arange(self.ensemble_size), size=1)[0]
            self._update_learning_rate(self.policy.optimizers[update_net_index])

            with th.no_grad():
                optimal_action = self.policy.nets[update_net_index](replay_data.next_observations).argmax(
                    dim=1).reshape(-1, 1)

                q_values_others = []
                for i in range(self.ensemble_size):
                    if i != update_net_index:
                        net_q_value = self.policy.nets[i](replay_data.observations)
                        net_q_value = th.gather(net_q_value, dim=1, index=optimal_action)
                        q_values_others.append(net_q_value)

                other_qs = th.hstack(q_values_others).sum(dim=1) / (self.ensemble_size - 1)
                other_qs = other_qs.reshape(-1, 1)

                update_net_qs = self.policy.nets[update_net_index](replay_data.observations)
                update_net_q_value = th.gather(update_net_qs, dim=1,
                                           index=replay_data.actions)
                target_loss = replay_data.rewards + (1 - replay_data.dones) * self.gamma * other_qs - update_net_q_value

            actual_q = self.policy.nets[i](replay_data.observations)
            actual_q = th.gather(actual_q, dim=1,
                                           index=replay_data.actions).reshape(-1, 1)
            loss = F.mse_loss(actual_q, target_loss)

            losses.append(loss.item())

            self.policy.optimizers[update_net_index].zero_grad()
            loss.backward()
            self.policy.optimizers[update_net_index].step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action = np.array([self.action_space.sample() for _ in range(n_batch)])
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state
