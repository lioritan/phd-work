from typing import Optional, Tuple, Type, Union, Callable, Dict, Any, List

from numpy.core._multiarray_umath import ndarray
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
import torch as th
import torch.nn.functional as F
import scipy.stats

from student_algorithms.pets.pets_policies import PETSPolicy


class PETS(OffPolicyAlgorithm):  # because replay buffer

    # MPC controller with CEM for selecting actions
    # Note: only works for continuous actions
    # Note: for serialization to work, all params need defaults
    def __init__(self,
                 policy: Union[Type[BasePolicy], str],
                 env: Union[GymEnv, str],
                 env_state_reward_func: Callable = None,
                 ensemble_size: int = 5,
                 mpc_horizon: int = 5,
                 num_iterations: int = 10,
                 num_candidates: int = 20,
                 num_elites: int = 5,
                 min_variance: float = 1e-3,
                 cem_alpha: float = 0.25,
                 num_particles: int = 20,
                 episodes_until_model_train: int = 5,

                 learning_rate: Union[float, Callable] = 1e-4,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 gradient_steps: int = 1,
                 n_episodes_rollout: int = 1,
                 action_noise: Optional[ActionNoise] = None,
                 optimize_memory_usage: bool = False,
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = "auto",
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 _init_setup_model= False,
                 ):
        super(PETS, self).__init__(policy,
                                   env,
                                   policy_base=PETSPolicy,
                                   learning_rate=learning_rate,
                                   buffer_size=buffer_size,
                                   learning_starts=0,
                                   batch_size=batch_size,
                                   tau=1,
                                   gamma=gamma,
                                   train_freq=-1,
                                   gradient_steps=gradient_steps,
                                   n_episodes_rollout=n_episodes_rollout,
                                   action_noise=action_noise,
                                   optimize_memory_usage=optimize_memory_usage,
                                   policy_kwargs=policy_kwargs,
                                   tensorboard_log=tensorboard_log,
                                   verbose=verbose,
                                   device=device,
                                   support_multi_env=False,
                                   create_eval_env=create_eval_env,
                                   monitor_wrapper=monitor_wrapper,
                                   seed=seed,
                                   use_sde=False,
                                   sde_sample_freq=-1,
                                   use_sde_at_warmup=False,
                                   sde_support=False,
                                   remove_time_limit_termination=True)

        self.ensemble_size = ensemble_size
        self.mpc_horizon = mpc_horizon
        self.num_particles = num_particles
        if policy_kwargs is None:
            self.policy_kwargs = {}
        self.policy_kwargs["ensemble_size"] = ensemble_size
        self.first_run = True
        self.eps = episodes_until_model_train
        self._setup_model()

        self.state_reward_func = env_state_reward_func
        self.cem = CEM(self.action_upper_bound, self.action_lower_bound, num_iterations, num_candidates, num_elites,
                       min_variance, cem_alpha)

    def _setup_model(self) -> None:
        super(PETS, self)._setup_model()
        self.action_upper_bound = self.env.action_space.high
        self.action_lower_bound = self.env.action_space.low
        self.existing_actions = [(self.action_upper_bound + self.action_lower_bound) / 2 for i in
                                 range(self.mpc_horizon)]
        self.action_variance = np.array([(self.action_upper_bound - self.action_lower_bound) ** 2 / 16 for i in
                                         range(self.mpc_horizon)])

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # parameters should be set so that this is called exactly once per n_episodes_rollout episodes
        if self.first_run:
            self.eps -= 1
            if self.eps <= 0:  # collect more rollouts
                return
            self.first_run = False

        # use replay buffer to fit a new dynamics model
        train_obs = th.tensor(self.replay_buffer.observations[:self.replay_buffer.size(), 0, :], device=self.device)
        train_actions = th.tensor(self.replay_buffer.actions[:self.replay_buffer.size(), 0, :], device=self.device)
        train_targets = th.tensor(self.replay_buffer.next_observations[:self.replay_buffer.size(), 0, :],
                                  device=self.device)

        for step in range(gradient_steps):
            for net in self.policy.models:
                inds = np.random.randint(0, train_obs.shape[0], size=(batch_size,))  # pick random indices for each net
                predicted_data = net(train_obs[inds, :], train_actions[inds, :])
                loss = F.mse_loss(predicted_data, train_targets[inds, :])

                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
        self._n_updates += gradient_steps

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # if not trained, random
        if self.first_run:
            return self.existing_actions[0].reshape(1, -1), state

        new_actions, action_variance = self.cem.get_new_actions(self.existing_actions, self.action_variance,
                                                                self.calculate_mpc_reward, observation)
        self.existing_actions = new_actions
        return new_actions[0].reshape(1, -1), state

    def calculate_mpc_reward(self, action_sequence, observation):
        particle_states = [np.copy(observation) for i in range(self.num_particles)]
        particle_rewards = [0 for i in range(self.num_particles)]
        for timestep in range(self.mpc_horizon):
            action = action_sequence[timestep]
            action_l2_norm = np.linalg.norm(action, 2)
            step_rewards = [self.state_reward_func(s) - action_l2_norm for s in particle_states]
            particle_rewards = [particle_rewards[i] + step_rewards[i] for i in range(len(particle_rewards))]
            particle_states = [self.policy.get_next(th.tensor(s, dtype=th.float32, device=self.device),
                                                    th.tensor(action.reshape(1, -1), dtype=th.float32, device=self.device))
                               for s in particle_states]
        return np.mean(particle_rewards)


class CEM:
    def __init__(self, upper_bound, lower_bound, num_iterations, num_candidates, num_elites, minimum_variance, alpha):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.min_variance = minimum_variance
        self.alpha = alpha

    def get_new_actions(self, current_actions, action_variance, reward_function, start_state):
        action_dim = current_actions[0].shape[0]
        horizon = len(current_actions)
        # normal(0, 1) between -2 and 2
        random_normal = scipy.stats.truncnorm(-2, 2,
                                              loc=np.zeros(action_dim * horizon), scale=np.ones(action_dim * horizon))

        curr_means = np.array(current_actions)
        curr_var = action_variance
        iter = 0
        while (iter < self.num_iterations) and np.max(curr_var) > self.min_variance:
            lb_dist, ub_dist = curr_means - self.lower_bound, self.upper_bound - curr_means
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), curr_var)

            candidates = random_normal.rvs(size=[self.num_candidates, action_dim * horizon]) * np.sqrt(
                constrained_var).reshape(-1) + np.array(curr_means).reshape(-1)
            candidate_costs = np.array(
                [reward_function(candidates[i, :].reshape(horizon, action_dim), start_state) for i in
                 range(self.num_candidates)])
            elites = candidates[np.argsort(candidate_costs)][:self.num_elites]

            curr_means = self.alpha * curr_means + (1 - self.alpha) * np.mean(elites, axis=0).reshape(horizon, action_dim)
            curr_var = self.alpha * curr_var + (1 - self.alpha) * np.var(elites, axis=0).reshape(horizon, action_dim)

            iter += 1
        return curr_means, curr_var
