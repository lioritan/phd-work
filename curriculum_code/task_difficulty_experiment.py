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


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task, hard_task, image_based=False):
    wrapper = ParamLeakingEnvWrapper(wrapper)  # TODO: does not work for maze
    random_teacher = RandomTeacher(None, wrapper)

    ref_env = wrapper.create_env(easy_task)

    # student = A2C(policy='MlpPolicy' if not image_based else "CnnPolicy", env=wrapper.create_env(easy_task), verbose=0)
    student = PPO(policy='MlpPolicy', env=ref_env,
                  verbose=0,
                  policy_kwargs={"net_arch": [dict(pi=[8, 8], vf=[8, 8])]},
                  n_steps=steps_per_task // 4)

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


def run_cartpole():
    easy_params = {
        "pole_length": 0.2,
        "cart_mass": 1.2,
        "pole_mass": 0.05
    }
    hard_params = {
        "pole_length": 0.8,
        "cart_mass": 0.8,
        "pole_mass": 0.2
    }
    measure_difficulty(300, 1000, CartpoleWrapper(), easy_params, hard_params)


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
    # measure_difficulty(400, 1000, LunarLanderWrapper(), easy_params, hard_params)
    measure_difficulty(400, 1000, LunarLanderWrapper(), easy_params, hard_params)


def run_walker():
    easy_params = {
        "climbing_surface_size": 0.0,
        "gap_pos": 10,
        "gap_size": 0.0,
        "obstacle_spacing": 10.0,
    }
    hard_params = {
        "climbing_surface_size": 1.0,
        "gap_pos": 5.0,
        "gap_size": 10.0,
        "obstacle_spacing": 5.0,
    }
    measure_difficulty(10000, 1000, WalkerWrapper(walker_type="classic_bipedal", walker_params={}), easy_params,
                       hard_params)


def run_custom_gridworld():
    easy_params = {
        "depth": 2,
        "width": 1,
        "keys": 0,
        "maze_percentage": 0.0,
    }
    hard_params = {
        "depth": 4,
        "width": 1,
        "keys": 6,
        "maze_percentage": 1.0,
    }
    measure_difficulty(5 * 5 * 4 * 4, 500, GridworldsCustomWrapper(), easy_params, hard_params, image_based=False)


if __name__ == "__main__":
    # run_cartpole()
    run_lunarlander()
    # run_custom_gridworld()
    # run_walker()
