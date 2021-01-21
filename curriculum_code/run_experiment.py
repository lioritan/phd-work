# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
from typing import Dict, Any

import gym
import procgen
from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.eval.history_metrics import plot_reward_graph, plot_diversity_graph, plot_tsne_task_distribution
from curriculum.teacher import Teacher
from curriculum.teachers.adr_teacher import AdrTeacher
from curriculum.teachers.learning_rate_sampling_teacher import LearningRateSamplingTeacher
from curriculum.teachers.learning_rate_teacher import LearningRateTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.riac_teacher import RiacTeacher
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

    eval_run(student, env)


def eval_run(student, env):
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

    #teacher = RandomTeacher(None, CartpoleWrapper())

    teacher = LearningRateTeacher({"alpha": 0.9, "epsilon": 0.1}, CartpoleWrapper())

    #teacher = LearningRateSamplingTeacher({"k": 5}, CartpoleWrapper())

    for i in range(200):
        teacher.train_k_actions(student, 200)
    #plot_reward_graph(teacher)
    #plot_diversity_graph(teacher)
    #plot_tsne_task_distribution(teacher)


def check_continuous():
    env = get_classic_walker().create_env({
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
        # "motors_torque": 80
    })
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=200)

    teacher = RiacTeacher({"max_region_size": 30}, get_classic_walker())

    #teacher = AdrTeacher({"reward_thr": 0, "initial_task": [0, 10, 3, 6]}, get_classic_walker())

    for i in range(100):
        teacher.train_k_actions(student, 400)

    plot_reward_graph(teacher)
    plot_diversity_graph(teacher)
    plot_tsne_task_distribution(teacher)
    eval_run(student, env)


#check_sanity()
check_continuous()

# l()
