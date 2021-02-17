# consider splitting some of the code here
import datetime
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, A2C, SAC, DQN
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
from environment.simple_envs.parametric_cartpole import CartpoleWrapper
from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper


def run_comparison(steps_per_task, tasks, wrapper, easy_task, hard_task, image_based=False):
    ref_env = wrapper.create_env(easy_task)
    i_task = [2, 1, 0, 0]

    teachers_list = [
        RandomTeacher(None, wrapper),
        RiacTeacher({"max_region_size": tasks // 10}, wrapper),
        AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                    "boundary_sampling_p": 0.7,
                    "reward_thr": 0.5,
                    "queue_len": 10}, wrapper),
        AlpGmmTeacher({"potential_ks": np.arange(2, 6, 1),
                       "warm_start": False,
                       "gmm_fitness_fun": "bic",
                       "nb_em_init": 2,
                       "fit_rate": 100,
                       "random_task_ratio": 0.2,
                       "alp_buffer_size": tasks
                       }, wrapper),

        # TODO: doesn't work on image obs
        # LearnedQShaping({"scale": 1.0,
        #                  "step_size": 0.01,
        #                  "obs_shape": ref_env.observation_space.shape[0],
        #                  "action_shape": 1, # TODO: does not work well, and also code bugs
        #                  "network_dimensions": [16, 16],
        #                  "discount": 0.99
        #                  },
        #                 wrapper, RandomTeacher(None, wrapper)),
        # LearnedVShaping({"scale": 1.0,
        #                  "step_size": 0.01,
        #                  "obs_shape": ref_env.observation_space.shape[0],
        #                  "network_dimensions": [16, 16],
        #                  "discount": 0.99
        #                  },
        #                 wrapper, RandomTeacher(None, wrapper)),
        LongEpisodeShaping({"scale": 0.1,
                            "is_strong": False
                            }, wrapper, RandomTeacher(None, wrapper)),

        ConstMixtureTeacher({"step_size": 0.01,
                             "teachers": [
                                 AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                                             "boundary_sampling_p": 0.7,
                                             "reward_thr": 0.5,
                                             "queue_len": 10}, wrapper),
                                 RandomTeacher(None, wrapper),
                             ]
                             }, wrapper),
        RewardStepsizeMixtureTeacher({
            "teachers": [
                AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                            "boundary_sampling_p": 0.7,
                            "reward_thr": 0.5,
                            "queue_len": 10}, wrapper),
                RandomTeacher(None, wrapper),
            ]
        }, wrapper),
        RewardMixtureTeacher({
            "teachers": [
                AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                            "boundary_sampling_p": 0.7,
                            "reward_thr": 0.5,
                            "queue_len": 10}, wrapper),
                RandomTeacher(None, wrapper),
            ]
        }, wrapper),
        PredictionMixtureTeacher({"network_dimensions": [8, 8],
                                  "teachers": [
                                      AdrTeacher({"initial_task": i_task,  # TODO: code cleanup
                                                  "boundary_sampling_p": 0.7,
                                                  "reward_thr": 0.5,
                                                  "queue_len": 10}, wrapper),
                                      RandomTeacher(None, wrapper),
                                  ]
                                  }, wrapper),
        PredictionMixtureTeacher({"regression": True,
                                  "teachers": [
                                      AdrTeacher({"initial_task": i_task,  # TODO: code cleanup,
                                                  "boundary_sampling_p": 0.7,
                                                  "reward_thr": 0.5,
                                                  "queue_len": 10}, wrapper),
                                      RandomTeacher(None, wrapper),
                                  ]
                                  }, wrapper),
    ]

    student_tasks = [easy_task, hard_task]

    baselines = [PredefinedTasksTeacher({"tasks": [student_task]}, wrapper) for student_task in student_tasks]

    eval_rewards = np.zeros((len(teachers_list) + 2, tasks, 2))

    sub_step = 10
    subsampled_task_range = list(range(0, tasks, sub_step))

    teacher_ind = 0
    for teacher in teachers_list + baselines:
        print(f"teacher {teacher_ind}")
        # student_agent = PPO(policy='MlpPolicy' if not image_based else "CnnPolicy", env=wrapper.create_env(easy_task),
        #                     verbose=0,
        #                     n_steps=steps_per_task // 4)
        student_agent = A2C(policy='MlpPolicy' if not image_based else "CnnPolicy", env=ref_env,
                            verbose=0)


        for i in tqdm(range(tasks)):  # tqdm adds a progress bar
            teacher.train_k_actions(student_agent, steps_per_task)
            if i not in subsampled_task_range:
                continue
            total_reward, _ = teacher.evaluate(steps_per_task, easy_task, student_agent)
            eval_rewards[teacher_ind, i, 0] = total_reward
            total_reward2, _ = teacher.evaluate(steps_per_task, hard_task, student_agent)
            eval_rewards[teacher_ind, i, 1] = total_reward2

        teacher_ind += 1

    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(f"./results/{date_string}/teachers/{wrapper.name}", exist_ok=True)
    with open(f"./results/{date_string}/teachers/{wrapper.name}/eval_rewards.pkl", "wb") as fptr:
        pickle.dump(eval_rewards, fptr)

    # for each teacher, reward for eval over time (3 graphs)
    for i, difficulty in enumerate(["easy", "hard"]):
        for j in range(len(teachers_list)):
            plt.plot(subsampled_task_range, eval_rewards[j, ::sub_step, i],
                     label=str(teachers_list[j].__class__.__name__))
        plt.plot(subsampled_task_range, eval_rewards[-2, ::sub_step, i], label="Train on Easy")
        plt.plot(subsampled_task_range, eval_rewards[-1, ::sub_step, i], label="Train on Hard")
        plt.xlabel('# tasks')
        plt.ylabel(f'mean eval reward - {difficulty}')
        plt.legend(loc='lower right', bbox_to_anchor=(1.05, 1))
        date_string = datetime.datetime.today().strftime('%Y-%m-%d')
        plt.savefig(f"./results/{date_string}/teachers/{wrapper.name}/eval_{difficulty}.jpg")
        plt.clf()


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


run_custom_gridworld()
#run_lunarlander()