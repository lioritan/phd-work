from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import numpy as np
import torch as th
from typing import Optional, Tuple, Type, Union, Callable, Dict, Any, List

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from tqdm import tqdm

from environment.simple_envs.parametric_lunarlander_continuous import LunarLanderContinuousWrapper
from student_algorithms.gp_model_ensemble.dpgpmm_policy import DPGPMMPolicy
from student_algorithms.gp_model_ensemble.mpc.mpc import MPC


class DPGPMMAlgorithm(OnPolicyAlgorithm):  # because replay buffer
    def __init__(self,
                 policy: Union[Type[BasePolicy], str],
                 env: Union[GymEnv, str],
                 env_state_reward_func: Callable = None,
                 mpc_horizon: int = 15,
                 num_iterations: int = 5,
                 num_candidates: int = 200,
                 num_elites: int = 20,
                 cem_alpha: float = 0.1,
                 num_particles: int = 1,
                 mm_alpha: float = 1.5,
                 mm_burnin: int = 5,
                 mm_merge_threshold: float = 10.0,
                 gp_iter: int = 5,
                 max_inducing_point: int = 1800,
                 trigger_induce: int = 2000,
                 sample_number: int = 50,
                 warm_up_time: int = 1000 * 5,

                 learning_rate: Union[float, Callable] = 1e-1,
                 n_steps: int = 1,  # steps until model update
                 gamma: float = 0.99,
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = "auto",
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 _init_setup_model=False,
                 ):
        super(DPGPMMAlgorithm, self).__init__(policy,
                                              env,
                                              learning_rate=learning_rate,
                                              n_steps=n_steps,
                                              gamma=gamma,
                                              gae_lambda=1.0,
                                              ent_coef=0,
                                              vf_coef=0,
                                              max_grad_norm=0,
                                              policy_kwargs=policy_kwargs,
                                              tensorboard_log=tensorboard_log,
                                              verbose=verbose,
                                              device=device,
                                              create_eval_env=create_eval_env,
                                              monitor_wrapper=monitor_wrapper,
                                              seed=seed,
                                              use_sde=False,
                                              sde_sample_freq=-1,
                                              _init_setup_model=False,
                                              )

        self.mpc_horizon = mpc_horizon
        self.num_particles = num_particles
        if policy_kwargs is None:
            self.policy_kwargs = {}
        self.policy_kwargs["alpha"] = mm_alpha
        self.policy_kwargs["merge_burnin"] = mm_burnin
        self.policy_kwargs["merge_threshold"] = mm_merge_threshold
        self.policy_kwargs["gp_iter"] = gp_iter
        self.policy_kwargs["max_inducing_point"] = max_inducing_point
        self.policy_kwargs["trigger_induce"] = trigger_induce
        self.policy_kwargs["sample_number"] = sample_number
        self._setup_model()

        self.state_reward_func = env_state_reward_func

        self.mpc = MPC({"optimizer": "CEM",
                        "CEM": {
                            "horizon": self.mpc_horizon,  # how long of the horizon to predict
                            "popsize": num_candidates,  # how many random samples for mpc
                            "particle": self.num_particles,  # number of particles to enlarge
                            "gamma": 1,  # reward discount coefficient
                            "action_low": env.action_space.low,  # lower bound of the solution space
                            "action_high": env.action_space.high,  # upper bound of the solution space
                            "action_dim": env.action_space.shape[0],
                            "max_iters": num_iterations,
                            "num_elites": num_elites,
                            "epsilon": 0.001,
                            "alpha": cem_alpha,
                            "init_mean": 0,
                            "init_var": 1,
                            "cost_function": self.state_reward_func,
                        }
                        })
        self.warm_up = True
        self.warm_up_steps_remaining = warm_up_time
        self.warm_up_buffer = []
        self.policy.forward = self.__wrap_policy_predict(lambda s: self.env.action_space.sample())

    def finish_warm_up(self):
        self.warm_up = False
        self.warm_up_buffer = None
        self.policy.forward = self.__wrap_policy_predict(lambda s:
                                                         self.mpc.act(None, self.policy.gpmm_model, s,
                                                                      ground_truth=False))

    def __wrap_policy_predict(self, action_predict_func):
        return lambda s: (
        th.tensor(action_predict_func(s.detach().cpu().numpy())).reshape(1, -1), th.zeros(1), th.zeros(1))

    def train(self) -> None:
        existing_data = self.rollout_buffer.get(batch_size=None)
        model_data = [(example.observations[0, :].cpu().numpy(),
                       example.actions[0, :].cpu().numpy(),
                       ) for example in existing_data]
        for i in range(len(model_data)):
            if i + 1 >= len(model_data):
                break
            model_data[i] = (model_data[i][0], model_data[i][1], model_data[i + 1][0] - model_data[i][0])
        last_obs = self._last_obs[0, :]
        model_data[-1] = (model_data[-1][0], model_data[-1][1], last_obs - model_data[-1][0])

        if not self.warm_up:
            for data_point in model_data:
                self.policy.gpmm_model.fit(data_point)
                return
        else:
            timesteps_performed = len(model_data)
            self.warm_up_steps_remaining -= timesteps_performed
            if self.warm_up_steps_remaining > 0:
                self.warm_up_buffer.extend(model_data)
            else:
                for data_point in self.warm_up_buffer:
                    self.policy.gpmm_model.fit(data_point)
                self.finish_warm_up()

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.mpc.act(None, self.policy.gpmm_model, observation, ground_truth=False), state

    def set_env(self, env) -> None:
        super(OnPolicyAlgorithm, self).set_env(env)
        reward_model = env.reward_model()
        # torch->numpy
        self.state_reward_func = lambda s, a: reward_model(s, a).cpu().numpy()
        self.mpc.set_cost_func(self.state_reward_func)
