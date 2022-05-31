# consider splitting some of the code here
import argparse
import datetime
import os
import pickle

import gym
from stable_baselines3 import PPO, A2C

import wandb
import numpy as np

from tqdm import tqdm

from curriculum.eval.task_difficulty_estimate import estimate_task_difficulties
from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.random_teacher import RandomTeacher
#from environment.parametric_mujoco.parametric_ant import AntWrapper
#from environment.parametric_mujoco.parametric_half_cheetah import HalfCheetahWrapper
from environment.parametric_mujoco.parametric_pendulum_locomotion import MBPendulumAngleContinuousWrapper
from student_algorithms.gp_model_ensemble.dpgpmm_algorithm import DPGPMMAlgorithm
from student_algorithms.gp_model_ensemble.dpgpmm_policy import DPGPMMPolicy
from student_algorithms.nn_model_ensemble.nnmm_algorithm import NNMMAlgorithm
from student_algorithms.nn_model_ensemble.nnmm_policy import NNMMPolicy


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task, student_alg="PPO"):
    random_teacher = RandomTeacher(None, wrapper)

    # random_teacher = PredefinedTasksTeacher({"tasks":
    #     [{"goal_angle": 0}]*100
    # }, wrapper)
    #random_teacher = PredefinedTasksTeacher({"tasks": [easy_task]}, wrapper)

    wandb.init(project=f'sched_{wrapper.name}', entity='liorf', save_code=True)
    config = wandb.config
    config.task = wrapper.name
    config.teacher = str(random_teacher)
    config.teacher_params = random_teacher.__dict__
    config.steps_per_task = steps_per_task
    config.num_tasks = tasks

    ref_env = wrapper.create_env(easy_task)

    if student_alg == "NN" or student_alg == "NN_MIX":
        student = NNMMAlgorithm(policy=NNMMPolicy, env=ref_env,
                                verbose=0,
                                env_state_reward_func=ref_env.reward_model(),
                                n_steps=1,
                                mm_burnin=20,
                                learning_rate=1e-2,
                                n_epochs=10,
                                is_res_net=False,
                                mpc_horizon=15,
                                optimizer="CEM",
                                policy_kwargs={"net_arch": [8, 8]},
                                is_mixed_model=(student_alg == "NN_MIX"),
                                warm_up_time=steps_per_task)  # Note: assumes all envs have a reward model
    elif student_alg == "GP":
        student = DPGPMMAlgorithm(policy=DPGPMMPolicy, env=ref_env,
                                  verbose=0,
                                  env_state_reward_func=ref_env.reward_model(),
                                  n_steps=1,
                                  mm_burnin=20,
                                  warm_up_time=steps_per_task)
    else:
        student = PPO(policy='MlpPolicy', env=ref_env,
                      verbose=0,
                      policy_kwargs={"net_arch": [dict(pi=[8, 8], vf=[8, 8])]},
                      n_steps=steps_per_task // 4)

        # student = A2C(policy='MlpPolicy', env=ref_env,
        #               policy_kwargs={"net_arch": [8, 8]},
        #                     verbose=0)

    config.student = str(student)
    config.student_params = student.__dict__

    wandb.watch(student.policy)  # TODO: put a model/policy (th module) and it logs gradients and model params

    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H%M') + student_alg
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)

    for i in tqdm(range(tasks)):
        random_teacher.train_k_actions(student, steps_per_task)
        wandb.log({"task_num": i,
                   "student_reward": random_teacher.history.history[-1][1]/steps_per_task,
                   "teacher_task": random_teacher.history.history[-1][0], })
        if i % 5 == 0 and i > 0:
            difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 10, 3, steps_per_task)
            avg_reward_per_step = difficulty_estimates/steps_per_task
            wandb.log({"task_num": i,
                       "reward_std": avg_reward_per_step.std(),
                       "mean_reward": avg_reward_per_step.mean(),
                       "total_reward": difficulty_estimates.mean(),
                       "task_rewards": avg_reward_per_step.mean(axis=1),
                       "best_subtask_reward": avg_reward_per_step.mean(axis=1).max(),
                       "best_subtask_index": avg_reward_per_step.mean(axis=1).argmax()})
            params_arr = np.array(task_params)
            param_names = [name for name in wrapper.parameters.keys()]
            for param_num in range(params_arr.shape[1]):
                wandb.log({"task_num": i,
                           f"task_reward_corrs_{param_names[param_num]}":
                               np.corrcoef(avg_reward_per_step.mean(axis=1), params_arr[:, param_num])[0, 1]})
            with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_{i}.pkl", "wb") as fptr:
                pickle.dump((difficulty_estimates, task_params), fptr)
            # TODO: save does not work for nnmm/dpgpmm, missing attributes
            # student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student_{i}.agent")

    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(random_teacher.history.history, fptr)

    difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 60, 3, steps_per_task)
    print(zip(difficulty_estimates.mean(axis=1).tolist(), task_params))

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


def run_cheetah(steps, tasks, student):
    easy_params = {
        "expected_speed": 1.1,
    }
    measure_difficulty(steps, tasks, HalfCheetahWrapper(), easy_params, student_alg=student)


def run_ant(steps, tasks, student):
    easy_params = {
        "goal_x": 1.1,
        "goal_y": 1.1
    }
    measure_difficulty(steps, tasks, AntWrapper(), easy_params, student_alg=student)


def run_pend(steps, tasks, student):
    easy_params = {
        "goal_angle": 1.5,
    }
    measure_difficulty(steps, tasks, MBPendulumAngleContinuousWrapper(), easy_params, student_alg=student)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--student', type=str, default='PPO',
                        help='GP/PPO/NN/NN_MIX')
    parser.add_argument('--env', type=str, default='pendulum',
                        help='pendulum/cheetah/ant')
    parser.add_argument('--tasks', type=int, default=50,
                        help='#tasks')
    parser.add_argument('--steps', type=int, default=1000,
                        help='#steps')
    args = parser.parse_args()

    if args.env == "pendulum":
        run_pend(args.steps, args.tasks, args.student)
    elif args.env == "cheetah":
        run_cheetah(args.steps, args.tasks, args.student)
    elif args.env == "ant":
        run_ant(args.steps, args.tasks, args.student)
