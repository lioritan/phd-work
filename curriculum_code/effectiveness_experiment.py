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

from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper


def run_effectiveness(steps_per_task, tasks, wrapper, easy_task, hard_task):
    random_teacher = RandomTeacher(None, wrapper)

    student_tasks = [easy_task, hard_task]

    repeating_custom_teacher = PredefinedTasksTeacher({"tasks": student_tasks}, wrapper)

    simple_teacher = RiacTeacher({"max_region_size": tasks // 10}, wrapper)

    baselines = [PredefinedTasksTeacher({"tasks": [student_task]}, wrapper) for student_task in student_tasks]

    eval_rewards = np.zeros((5, tasks, 2))

    teacher_ind = 0
    for teacher in [random_teacher, repeating_custom_teacher, simple_teacher] + baselines:
        print(f"teacher {teacher_ind}")
        student_agent = PPO(policy='MlpPolicy', env=wrapper.create_env(easy_task), verbose=0,
                            n_steps=steps_per_task // 4)

        for i in tqdm(range(tasks)):  # tqdm adds a progress bar
            teacher.train_k_actions(student_agent, steps_per_task)
            total_reward, _ = teacher.evaluate(steps_per_task, easy_task, student_agent)
            eval_rewards[teacher_ind, i, 0] = total_reward
            total_reward, _ = teacher.evaluate(steps_per_task, hard_task, student_agent)
            eval_rewards[teacher_ind, i, 1] = total_reward

        teacher_ind += 1

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/effectiveness/{wrapper.name}", exist_ok=True)

    plot_diversity_graph(simple_teacher, fname=f"./results/{date_string}/effectiveness/{wrapper.name}/diversity.jpg",
                         continuous_sensativity=0.1)
    plt.clf()

    sub_step = 10
    subsampled_task_range = list(range(0, tasks, sub_step))

    # plot train rewards for each teacher
    plt.plot(subsampled_task_range, [x[1] for x in random_teacher.history[::sub_step]], label="random teacher")
    plt.plot(subsampled_task_range, [x[1] for x in repeating_custom_teacher.history[::sub_step]],
             label="custom teacher")
    plt.plot(subsampled_task_range, [x[1] for x in simple_teacher.history[::sub_step]], label="RIAC teacher")
    plt.plot(subsampled_task_range, [x[1] for x in baselines[0].history[::sub_step]], label="Train on Easy")
    plt.plot(subsampled_task_range, [x[1] for x in baselines[1].history[::sub_step]], label="Train on Hard")
    plt.xlabel('# tasks')
    plt.ylabel('mean train reward')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    plt.savefig(f"./results/{date_string}/effectiveness/{wrapper.name}/train.jpg")
    plt.clf()

    # for each teacher, reward for eval over time (3 graphs)
    for i, difficulty in enumerate(["easy", "hard"]):
        plt.plot(subsampled_task_range, eval_rewards[0, ::sub_step, i], label="random teacher")
        plt.plot(subsampled_task_range, eval_rewards[1, ::sub_step, i], label="custom teacher")
        plt.plot(subsampled_task_range, eval_rewards[2, ::sub_step, i], label="RIAC teacher")
        plt.plot(subsampled_task_range, eval_rewards[3, ::sub_step, i], label="Train on Easy")
        plt.plot(subsampled_task_range, eval_rewards[4, ::sub_step, i], label="Train on Hard")
        plt.xlabel('# tasks')
        plt.ylabel(f'mean eval reward - {difficulty}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        date_string = datetime.datetime.today().strftime('%Y-%m-%d')
        plt.savefig(f"./results/{date_string}/effectiveness/{wrapper.name}/eval_{difficulty}.jpg")
        plt.clf()


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
    run_effectiveness(300, 250, CartpoleWrapper(), easy_params, hard_params)


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
    run_effectiveness(400, 500, LunarLanderWrapper(), easy_params, hard_params)  # TODO


#run_cartpole()
run_lunarlander()
