# consider splitting some of the code here
import datetime
import os
import pickle
import random

import bokeh
import bokeh.plotting
import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import Spectral11
from stable_baselines3 import PPO, A2C, SAC, DQN, TD3, DDPG, HER
from stable_baselines3.common.noise import NormalActionNoise
from tqdm import tqdm

from curriculum.eval.history_metrics import plot_diversity_graph
from curriculum.teachers.adr_teacher import AdrTeacher
from curriculum.teachers.again_teacher import AgainTeacher
from curriculum.teachers.alp_gmm_teacher import AlpGmmTeacher
from curriculum.teachers.mixture_teachers.const_change_mixture_teacher import ConstMixtureTeacher
from curriculum.teachers.mixture_teachers.predicted_value_mixture_teacher import PredictionMixtureTeacher
from curriculum.teachers.mixture_teachers.reward_stepsize_mixture_teacher import RewardStepsizeMixtureTeacher
from curriculum.teachers.mixture_teachers.reward_weighted_mixture_teacher import RewardMixtureTeacher
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.reward_shaping.learned_q_shaping import LearnedQShaping
from curriculum.teachers.reward_shaping.learned_v_shaping import LearnedVShaping
from curriculum.teachers.reward_shaping.long_episode_shaping import LongEpisodeShaping
from curriculum.teachers.riac_teacher import RiacTeacher
from environment.gridworld_advanced.parametric_gridworld_advanced import GridworldsCustomWrapper
from environment.gridworld_advanced.parametric_gridworld_randomized import GridworldsRandomizedWrapper
from environment.parametric_walker_env.parametric_walker_wrapper import WalkerWrapper
from environment.simple_envs.gridworld_key_dynamic_difficulty import GridworldKeyWrapper
from environment.simple_envs.parametric_cartpole import CartpoleWrapper
from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper


def run_comparison(steps_per_task, tasks, wrapper, easy_task, hard_task, image_based=False):
    ref_env = wrapper.create_env(easy_task)
    i_task = [2, 1, 0, 0]

    teachers_list = [
        RandomTeacher(None, wrapper),
        AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                    "boundary_sampling_p": 0.7,
                    "reward_thr": 0.5,
                    "queue_len": 10}, wrapper),
    ]

    student_tasks = [easy_task, hard_task]

    baselines = [PredefinedTasksTeacher({"tasks": [student_task]}, wrapper) for student_task in student_tasks]

    students_list = [
        DQN(policy='MlpPolicy' if not image_based else "CnnPolicy", env=ref_env, learning_starts=200,
            tau=0.8,
            exploration_fraction=1.0, exploration_initial_eps=0.05, exploration_final_eps=0.05,
            verbose=0),
        DQN(policy='MlpPolicy' if not image_based else "CnnPolicy", env=ref_env, learning_starts=200,
            tau=0.8,
            exploration_fraction=1.0, exploration_initial_eps=1.0, exploration_final_eps=0.05,
            verbose=0),
        DQN(policy='MlpPolicy' if not image_based else "CnnPolicy", env=ref_env, learning_starts=200,
            tau=0.8,
            exploration_fraction=1.0, exploration_initial_eps=0.3, exploration_final_eps=0.3,
            verbose=0),
    ]

    eval_rewards = np.zeros((len(teachers_list) + 2, len(students_list), tasks, 2))

    sub_step = 10
    subsampled_task_range = list(range(0, tasks, sub_step))

    for teacher_ind, teacher in enumerate(teachers_list + baselines):
        print(f"teacher {teacher_ind}")
        for student_index, student_agent in enumerate(students_list):
            print(f"student {student_index}")
            for i in tqdm(range(tasks)):  # tqdm adds a progress bar
                teacher.train_k_actions(student_agent, steps_per_task)
                if i not in subsampled_task_range:
                    continue
                total_reward, _ = teacher.evaluate(steps_per_task, easy_task, student_agent)
                eval_rewards[teacher_ind, student_index, i, 0] = total_reward
                total_reward2, _ = teacher.evaluate(steps_per_task, hard_task, student_agent)
                eval_rewards[teacher_ind, student_index, i, 1] = total_reward2

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/exploration/{wrapper.name}", exist_ok=True)
    with open(f"./results/{date_string}/exploration/{wrapper.name}/eval_rewards.pkl", "wb") as fptr:
        pickle.dump(eval_rewards, fptr)

    # for each teacher, reward for eval over time (3 graphs)
    for i, difficulty in enumerate(["easy", "hard"]):
        for j in range(len(teachers_list)+2):
            t_name = str((teachers_list + baselines)[j].__class__.__name__)
            p = bokeh.plotting.figure(plot_width=1000, plot_height=600)
            for k in range(len(students_list)):
                p.line(subsampled_task_range, eval_rewards[j, k, ::sub_step, i], line_width=2, color=Spectral11[k],
                       alpha=0.8,
                       legend_label=str(students_list[k].exploration_initial_eps))
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            bokeh.plotting.output_file(
                f"./results/{date_string}/exploration/{wrapper.name}/eval_{difficulty}_{t_name}.html",
                title=f"{difficulty}-{t_name}")
            bokeh.plotting.save(p)


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
    run_comparison(5 * 5 * 4 * 4, 1000, GridworldsCustomWrapper(), easy_params, hard_params, image_based=False)


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
        "main_engine_power": 50.0,
        "side_engine_power": 10.0,
    }
    run_comparison(400, 1000, LunarLanderWrapper(), easy_params, hard_params)

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
    run_comparison(32 * 32 * 4, 500, GridworldsRandomizedWrapper(), easy_params, hard_params, image_based=False)

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
    run_comparison(300, 250, CartpoleWrapper(), easy_params, hard_params)


def run_grid_key():
    easy_params = {
        "difficulty": 1,
    }
    hard_params = {
        "difficulty": 6,
    }
    run_comparison(1000, 1000, GridworldKeyWrapper(), easy_params, hard_params)

#run_cartpole()
#run_custom_gridworld()
#run_lunarlander()
#run_random_gridworld()
run_grid_key()