# consider splitting some of the code here
from typing import Dict, Any

import gym
import procgen
from tqdm import tqdm
import os
import datetime
from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.eval.history_metrics import plot_reward_graph, plot_diversity_graph, plot_tsne_task_distribution, \
    plot_eval_performance, plot_eval_to_pretrain_performance
from curriculum.teacher import Teacher
from curriculum.teachers.adr_teacher import AdrTeacher
from curriculum.teachers.again_teacher import AgainTeacher
from curriculum.teachers.alp_gmm_teacher import AlpGmmTeacher
from curriculum.teachers.learning_rate_sampling_teacher import LearningRateSamplingTeacher
from curriculum.teachers.learning_rate_teacher import LearningRateTeacher
from curriculum.teachers.mixture_teachers.const_change_mixture_teacher import ConstMixtureTeacher
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher

from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.riac_teacher import RiacTeacher
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_walker_env.bodies.BodyTypesEnum import BodyTypesEnum
from environment.parametric_walker_env.parametric_continuous_flat_parkour import ParametricContinuousWalker
from environment.parametric_walker_env.parametric_walker_wrapper import get_classic_walker
from environment.simple_envs.parametric_cartpole import CartpoleWrapper
import numpy as np
import matplotlib.pyplot as plt


def run_basic_effectiveness_experiment(steps_per_task, tasks):
    """
    Run env with teacher vs no teacher
    :param steps_per_task:
    :param tasks:
    :return:
    """

    run_parametric_cartpole(steps_per_task, tasks)

    pass

    # TODO: try and pick easy, medium, hard task for each env
    # TODO: random teacher, const teacher on the task, riac/adr
    # TODO: plot: task perf after every K tasks
    # maybe: effect of steps per iter? would be interesting to see for T total steps what the best division is
    # TODO: saving, checkpointing


def run_parametric_cartpole(steps_per_task, tasks):
    env1_wrapper = CartpoleWrapper()
    easy_params = {
        "pole_length": 0.2,
        "cart_mass": 1.2,
        "pole_mass": 0.05
    }
    task_easy = env1_wrapper.create_env(easy_params)  # easy?
    medium_params = {
        "pole_length": 0.5,
        "cart_mass": 1.2,
        "pole_mass": 0.1
    }
    task_medium = env1_wrapper.create_env(medium_params)  # medium
    hard_params = {
        "pole_length": 0.8,
        "cart_mass": 0.8,
        "pole_mass": 0.2
    }
    task_hard = env1_wrapper.create_env(hard_params)  # hard

    random_teacher = RandomTeacher(None, env1_wrapper)

    student_tasks = [easy_params, medium_params, hard_params]

    repeating_custom_teacher = PredefinedTasksTeacher({"tasks": student_tasks}, env1_wrapper)

    simple_teacher = RiacTeacher({"max_region_size": tasks // 10}, env1_wrapper)

    baselines = [PredefinedTasksTeacher({"tasks": [student_task]}, env1_wrapper) for student_task in student_tasks]

    eval_rewards = np.zeros((6, tasks, 3))

    teacher_ind = 0
    for teacher in [random_teacher, repeating_custom_teacher, simple_teacher] + baselines:
        print(f"teacher {teacher_ind}")
        student_agent = PPO(policy='MlpPolicy', env=task_easy, verbose=0, n_steps=steps_per_task // 4)

        for i in tqdm(range(tasks)):  # tqdm adds a progress bar
            teacher.train_k_actions(student_agent, steps_per_task)
            total_reward, _ = teacher.evaluate(steps_per_task, easy_params, student_agent)
            eval_rewards[teacher_ind, i, 0] = total_reward
            total_reward, _ = teacher.evaluate(steps_per_task, medium_params, student_agent)
            eval_rewards[teacher_ind, i, 1] = total_reward
            total_reward, _ = teacher.evaluate(steps_per_task, hard_params, student_agent)
            eval_rewards[teacher_ind, i, 2] = total_reward

        teacher_ind += 1

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/effectiveness/cartpole", exist_ok=True)

    plot_diversity_graph(simple_teacher, fname=f"./results/{date_string}/effectiveness/cartpole/diversity.jpg",
                         continuous_sensativity=0.1)
    plt.clf()
    # plot train rewards for each teacher
    plt.plot(range(tasks), [x[1] for x in random_teacher.history.history], label="random teacher")
    plt.plot(range(tasks), [x[1] for x in repeating_custom_teacher.history.history], label="custom teacher")
    plt.plot(range(tasks), [x[1] for x in simple_teacher.history.history], label="RIAC teacher")
    plt.plot(range(tasks), [x[1] for x in baselines[0].history.history], label="Train on Easy")
    plt.plot(range(tasks), [x[1] for x in baselines[1].history.history], label="Train on Medium")
    plt.plot(range(tasks), [x[1] for x in baselines[2].history.history], label="Train on Hard")
    plt.xlabel('# tasks')
    plt.ylabel('mean train reward')
    plt.legend(loc='upper right', bbox_to_anchor=(2, 1))
    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/effectiveness/cartpole", exist_ok=True)
    plt.savefig(f"./results/{date_string}/effectiveness/cartpole/train.jpg")
    plt.clf()

    # for each teacher, reward for eval over time (3 graphs)
    for i, difficulty in enumerate(["easy", "medium", "hard"]):
        plt.plot(range(tasks), eval_rewards[0, :, i], label="random teacher")
        plt.plot(range(tasks), eval_rewards[1, :, i], label="custom teacher")
        plt.plot(range(tasks), eval_rewards[2, :, i], label="RIAC teacher")
        plt.plot(range(tasks), eval_rewards[3, :, i], label="Train on Easy")
        plt.plot(range(tasks), eval_rewards[4, :, i], label="Train on Medium")
        plt.plot(range(tasks), eval_rewards[5, :, i], label="Train on Hard")
        plt.xlabel('# tasks')
        plt.ylabel(f'mean eval reward - {difficulty}')
        plt.legend(loc='upper right', bbox_to_anchor=(2, 1))
        date_string = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(f"./results/{date_string}/effectiveness/cartpole", exist_ok=True)
        plt.savefig(f"./results/{date_string}/effectiveness/cartpole/eval_{difficulty}.jpg")
        plt.clf()


run_basic_effectiveness_experiment(300, 1000)
