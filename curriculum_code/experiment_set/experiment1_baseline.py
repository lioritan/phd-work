# consider splitting some of the code here
import argparse
import datetime
import os
import pickle

import gym
from stable_baselines3 import PPO

import wandb
import numpy as np
import random
import torch as th

from tqdm import tqdm

from curriculum.eval.task_difficulty_estimate import estimate_task_difficulties
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from environment.parametric_mujoco.parametric_pendulum_locomotion import MBPendulumAngleContinuousWrapper
from student_algorithms.nn_model_ensemble.nnmm_algorithm import NNMMAlgorithm
from student_algorithms.nn_model_ensemble.nnmm_policy import NNMMPolicy


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task, student_alg="PPO", teacher_alg="random"):

    if teacher_alg == "random":
        teacher = RandomTeacher(None, wrapper)
    elif teacher_alg == "pi":
        teacher = PredefinedTasksTeacher({"tasks": [
                                                       {"goal_angle": np.pi}
                                                   ] * tasks}, wrapper)
    elif teacher_alg == "pi-half":
        teacher = PredefinedTasksTeacher({"tasks": [
                                                       {"goal_angle": np.pi/2}
                                                   ] * tasks}, wrapper)
    elif teacher_alg == "0":
        teacher = PredefinedTasksTeacher({"tasks": [
                                                       {"goal_angle": 0}
                                                   ] * tasks}, wrapper)

    wandb.init(project=f'exp1_baseline_pendulum', entity='liorf', save_code=True)
    config = wandb.config
    config.task = wrapper.name
    config.teacher = str(teacher_alg)
    config.teacher_params = teacher.__dict__
    config.steps_per_task = steps_per_task
    config.num_tasks = tasks

    ref_env = wrapper.create_env(easy_task)

    if student_alg == "NN":
        student = NNMMAlgorithm(policy=NNMMPolicy, env=ref_env,
                                verbose=0,
                                env_state_reward_func=ref_env.reward_model(),
                                n_steps=1,
                                mm_burnin=20,
                                learning_rate=1e-2,
                                n_epochs=10,
                                is_res_net=False,
                                mpc_horizon=10,
                                optimizer="CEM",
                                policy_kwargs={"net_arch": [8, 8]},
                                is_mixed_model=False,
                                warm_up_time=steps_per_task)  # Note: assumes all envs have a reward model
    else:
        student = PPO(policy='MlpPolicy', env=ref_env,
                      verbose=0,
                      policy_kwargs={"net_arch": [dict(pi=[8, 8], vf=[8, 8])]},
                      n_steps=steps_per_task // 5)

    config.student = str(student)
    config.student_params = student.__dict__

    wandb.watch(student.policy)  # TODO: make this do something for my method

    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H%M') + student_alg
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)

    for i in tqdm(range(tasks)):
        teacher.train_k_actions(student, steps_per_task)
        wandb.log({"task_num": i,
                   "student_reward": teacher.history.history[-1][1],
                   "teacher_task": teacher.history.history[-1][0], })
        if i > 0:
            wandb.log({"task_num": i,
                       "ALP": abs(teacher.history.history[-1][1] - teacher.history.history[-2][1]),
                       "LP": teacher.history.history[-1][1] - teacher.history.history[-2][1]})
        if i % 5 == 0 and i > 0:
            difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 10, 3, steps_per_task)
            wandb.log({"task_num": i,
                       "reward_std": difficulty_estimates.std(),
                       "mean_reward": difficulty_estimates.mean(),
                       "best_subtask_reward": difficulty_estimates.mean(axis=1).max(),
                       "best_subtask_index": difficulty_estimates.mean(axis=1).argmax()})
            params_arr = np.array(task_params)
            normalized_angles = np.abs(((params_arr+np.pi) % (2*np.pi)) - np.pi)
            wandb.log({"task_num": i,
                        "task_reward_corrs_angle":
                            np.corrcoef(difficulty_estimates.mean(axis=1), normalized_angles)[0, 1]})
            with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_{i}.pkl", "wb") as fptr:
                pickle.dump((difficulty_estimates, task_params), fptr)
            # TODO: save does not work for nnmm/dpgpmm, missing attributes
            # student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student_{i}.agent")

    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(teacher.history.history, fptr)

    # eval and record video
    wandb.gym.monitor()  # Any env used with gym wrapper monitor will now be recorded
    evaluate(steps_per_task, wrapper.create_env(easy_task),
             student, f"./results/{date_string}/difficulty/{wrapper.name}/")


def evaluate(action_limit, base_env, student, base_dir):
    eval_env = gym.wrappers.Monitor(base_env, directory=base_dir)
    student.set_env(eval_env)
    s = eval_env.reset()
    total_reward = 0
    episode_length = 0
    for i in range(action_limit):
        a, _ = student.predict(observation=s, deterministic=True)
        s, r, d, _ = eval_env.step(a)
        episode_length += 1
        total_reward += r
        if d == 1:
            break
    return total_reward, episode_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--student', type=str, default='PPO',
                        help='PPO/NN')
    parser.add_argument('--tasks', type=int, default=50,
                        help='#tasks')
    parser.add_argument('--steps', type=int, default=1000,
                        help='#steps')
    parser.add_argument('--schedule', type=str, default='random',
                        help='random/pi/0/pi-half')
    parser.add_argument('--seed', type=int, default=42,
                        help='#steps')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    easy_params = {
        "goal_angle": 1.5,
    }
    measure_difficulty(args.steps, args.tasks,
                       MBPendulumAngleContinuousWrapper(randomize_start=False, goal_in_state=True),
                       easy_params,
                       student_alg=args.student,
                       teacher_alg=args.schedule)

