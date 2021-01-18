# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
from typing import Dict, Any

import gym
import procgen
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.teacher import Teacher
from curriculum.teachers.random_teacher import RandomTeacher
from environment.environment_wrapper import EnvironmentWrapper


class MyEnvWrapper(EnvironmentWrapper):
    def create_env(self, parameter_values: Dict[str, Any]):
        return gym.make("CartPole-v1")


def check_sanity():
    env_id = "CartPole-v1"
    student = PPO(policy='MlpPolicy', env=env_id, batch_size=100, verbose=0, n_steps=200)

    #student = A2C(policy='MlpPolicy', env=env_id, verbose=1, n_steps=200)

    #student = DQN(policy='MlpPolicy', env=env_id, verbose=0, learning_starts=0, buffer_size=10000)

    teacher = RandomTeacher(None, MyEnvWrapper(env_id, {}))
    for i in range(101):
        teacher.train_k_actions(student, 200)

check_sanity()
