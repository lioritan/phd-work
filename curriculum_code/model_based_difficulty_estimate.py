# consider splitting some of the code here
import datetime
import os
import random
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, A2C
from tqdm import tqdm

from curriculum.eval.history_metrics import plot_diversity_graph
from curriculum.eval.task_difficulty_estimate import estimate_task_difficulties, plot_estimated_task_difficulties
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.riac_teacher import RiacTeacher
from environment.environment_wrapper import EnvironmentWrapper
from environment.gridworld_advanced.parametric_gridworld_advanced import GridworldsCustomWrapper
from environment.gridworld_advanced.parametric_gridworld_randomized import GridworldsRandomizedWrapper
from environment.parametric_walker_env.parametric_walker_wrapper import WalkerWrapper
from environment.simple_envs.parametric_cartpole import CartpoleWrapper
from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper

import stable_baselines3.ppo.policies as policies

from environment.simple_envs.parametric_lunarlander_continuous import LunarLanderContinuousWrapper
from student_algorithms.pets.pets import PETS
from student_algorithms.pets.pets_policies import PETSPolicy


class ParamLeakingWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, env_params):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low[0], env.observation_space.high[0],
                                                (env.observation_space.shape[0] + len(env_params),))
        self.param_values = env_params

    def observation(self, obs):
        return np.hstack((obs, self.param_values))


class ParamLeakingEnvWrapper(EnvironmentWrapper):
    def __init__(self, env):
        super().__init__()
        self.name = env.name
        self.parameters = env.parameters
        self._env = env

    def create_env(self, parameter_values):
        env_values = np.array([parameter_values[k] for k in self.parameters.keys()])
        return ParamLeakingWrapper(self._env.create_env(parameter_values), env_values)


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task, state_func):
    wrapper = ParamLeakingEnvWrapper(wrapper)  # TODO: does not work for maze
    random_teacher = RandomTeacher(None, wrapper)

    ref_env = wrapper.create_env(easy_task)

    student = PETS(policy=PETSPolicy, env=ref_env,
                   verbose=0,
                   policy_kwargs={"net_arch": [8, 8]},
                   env_state_reward_func=state_func,
                   ensemble_size=5,
                   mpc_horizon=3,
                   num_iterations=5,
                   num_candidates=10,
                   num_elites=3,
                   n_episodes_rollout=2)

    for i in tqdm(range(tasks)):
        random_teacher.train_k_actions(student, steps_per_task)
    difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 10, 3, steps_per_task)

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)
    with open(f"./results/{date_string}/difficulty/{wrapper.name}/data.pkl", "wb") as fptr:
        pickle.dump((difficulty_estimates, task_params), fptr)
    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(random_teacher.history.history, fptr)
    student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student.agent")
    plot_estimated_task_difficulties(difficulty_estimates, task_params,
                                     filename=f"./results/{date_string}/difficulty/{wrapper.name}/embedded_difficulties.png")


def run_lunarlander():
    easy_params = {
        "leg_height": 20,
        "leg_width": 5,
        "main_engine_power": 50.0,
        "side_engine_power": 10.0,
    }
    hard_params = {
        "leg_height": 40,
        "leg_width": 2,
        "main_engine_power": 10.0,
        "side_engine_power": 1.0,
    }
    measure_difficulty(400, 1000, LunarLanderContinuousWrapper(), easy_params, hard_params)


if __name__ == "__main__":
    run_lunarlander()
