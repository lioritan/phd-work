# consider splitting some of the code here
import datetime
import os
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, A2C
from tqdm import tqdm

from curriculum.eval.history_metrics import plot_diversity_graph
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.riac_teacher import RiacTeacher
from environment.gridworld_advanced.parametric_gridworld_advanced import GridworldsCustomWrapper
from environment.gridworld_advanced.parametric_gridworld_randomized import GridworldsRandomizedWrapper
from environment.parametric_walker_env.parametric_walker_wrapper import WalkerWrapper
from environment.simple_envs.parametric_cartpole import CartpoleWrapper
from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper


def run_effectiveness(steps_per_task, tasks, wrapper, easy_task, hard_task, image_based=False):
    random_teacher = RandomTeacher(None, wrapper)

    student_tasks = [easy_task, hard_task]

    repeating_custom_teacher = PredefinedTasksTeacher({"tasks": student_tasks}, wrapper)

    simple_teacher = RiacTeacher({"max_region_size": tasks // 10}, wrapper)

    baselines = [PredefinedTasksTeacher({"tasks": [student_task]}, wrapper) for student_task in student_tasks]

    eval_rewards = np.zeros((5, tasks, 2))

    teacher_ind = 0
    for teacher in [random_teacher, repeating_custom_teacher, simple_teacher] + baselines:
        print(f"teacher {teacher_ind}")
        # student_agent = PPO(policy='MlpPolicy' if not image_based else "CnnPolicy", env=wrapper.create_env(easy_task),
        #                     verbose=0,
        #                     n_steps=steps_per_task // 4)
        student_agent = A2C(policy='MlpPolicy' if not image_based else "CnnPolicy", env=wrapper.create_env(easy_task),
                            verbose=0)

        for i in tqdm(range(tasks)):  # tqdm adds a progress bar
            teacher.train_k_actions(student_agent, steps_per_task)
            total_reward, _ = teacher.evaluate(steps_per_task, easy_task, student_agent)
            eval_rewards[teacher_ind, i, 0] = total_reward
            total_reward, _ = teacher.evaluate(steps_per_task, hard_task, student_agent)
            eval_rewards[teacher_ind, i, 1] = total_reward

        teacher_ind += 1

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/effectiveness/{wrapper.name}", exist_ok=True)

    with open(f"./results/{date_string}/effectiveness/{wrapper.name}/teacher_histories.pkl", "wb") as fptr:
        pickle.dump(([x for x in random_teacher.history],
                     [x for x in repeating_custom_teacher.history],
                     [x for x in simple_teacher.history],
                     [x for x in baselines[0].history],
                     [x for x in baselines[1].history]), fptr)
    with open(f"./results/{date_string}/effectiveness/{wrapper.name}/eval_rewards.pkl", "wb") as fptr:
        pickle.dump(eval_rewards, fptr)
    plot_diversity_graph(simple_teacher, fname=f"./results/{date_string}/effectiveness/{wrapper.name}/diversity.jpg",
                         continuous_sensativity=1.0)
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
    plt.legend(loc='lower right', bbox_to_anchor=(1.05, 1))
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
        plt.legend(loc='lower right', bbox_to_anchor=(1.05, 1))
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
    run_effectiveness(400, 1000, LunarLanderWrapper(), easy_params, hard_params)


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
    run_effectiveness(10000, 500, WalkerWrapper(walker_type="classic_bipedal", walker_params={}), easy_params,
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
    run_effectiveness(5 * 5 * 4 * 4, 500, GridworldsCustomWrapper(), easy_params, hard_params, image_based=False)


def run_random_gridworld():
    easy_params = {
        "start_pos": 0,
        "goal_pos": 42,
    }
    for i in range(32 * 32):
        easy_params[f"pos {i}"] = 1
    hard_params = {
        "start_pos": 0,
        "goal_pos": 32 * 32 - 1,
    }
    random.seed(12)  # solvable maze
    for i in range(32 * 32):
        hard_params[f"pos {i}"] = random.choice([1, 1, 1, 1, 2, 9])  # 66% empty, 33% obstacles
    run_effectiveness(32 * 32 * 4, 500, GridworldsRandomizedWrapper(), easy_params, hard_params, image_based=False)


run_cartpole()
run_lunarlander()
run_custom_gridworld()
run_random_gridworld()
run_walker()
