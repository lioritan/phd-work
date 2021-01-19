# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
from typing import Dict, Any

import gym
import procgen
from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.teacher import Teacher
from curriculum.teachers.learning_rate_sampling_teacher import LearningRateSamplingTeacher
from curriculum.teachers.learning_rate_teacher import LearningRateTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from environment.environment_wrapper import EnvironmentWrapper
from environment.simple_envs.parametric_cartpole import CartpoleWrapper


def check_sanity():
    env_id = "CartPole-v1"
    student = PPO(policy='MlpPolicy', env=env_id, batch_size=100, verbose=0, n_steps=200)

    #student = A2C(policy='MlpPolicy', env=env_id, verbose=0, n_steps=200)

    #student = DQN(policy='MlpPolicy', env=env_id, verbose=0, learning_starts=0, buffer_size=10000)

    #teacher = RandomTeacher(None, CartpoleWrapper())

    #teacher = LearningRateTeacher({"alpha": 0.9, "epsilon": 0.1}, CartpoleWrapper())

    teacher = LearningRateSamplingTeacher({"k": 5}, CartpoleWrapper())

    for i in range(300):
        teacher.train_k_actions(student, 200)

check_sanity()
