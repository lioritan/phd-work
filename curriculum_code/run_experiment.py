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
from environment.parametric_walker_env.bodies.BodyTypesEnum import BodyTypesEnum
from environment.parametric_walker_env.parametric_continuous_flat_parkour import ParametricContinuousWalker
from environment.parametric_walker_env.parametric_walker_wrapper import get_classic_walker
from environment.simple_envs.parametric_cartpole import CartpoleWrapper


def learn_parameteric():
    env = get_classic_walker().create_env({
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
        "motors_torque": 80
    })
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=1024)
    student.learn(20000)
    s = env.reset()

    for i in range(200):
        a, _ = student.predict(observation=s)
        env.render()
        s, r, d, _ = env.step(a)
        if d == 1:
            print(r)
            return
    print(r)


def check_sanity():
    env_id = "CartPole-v1"
    student = PPO(policy='MlpPolicy', env=env_id, batch_size=100, verbose=0, n_steps=200)

    # student = A2C(policy='MlpPolicy', env=env_id, verbose=0, n_steps=200)

    # student = DQN(policy='MlpPolicy', env=env_id, verbose=0, learning_starts=0, buffer_size=10000)

    # teacher = RandomTeacher(None, CartpoleWrapper())

    # teacher = LearningRateTeacher({"alpha": 0.9, "epsilon": 0.1}, CartpoleWrapper())

    teacher = LearningRateSamplingTeacher({"k": 5}, CartpoleWrapper())

    for i in range(300):
        teacher.train_k_actions(student, 200)


# check_sanity()

learn_parameteric()
