# consider splitting some of the code here
import datetime
import os
import pickle

import gym
import wandb
import numpy as np

from tqdm import tqdm

from curriculum.eval.task_difficulty_estimate import estimate_task_difficulties
from curriculum.teachers.random_teacher import RandomTeacher
from environment.parametric_mujoco.parametric_ant import AntWrapper
from environment.parametric_mujoco.parametric_half_cheetah import HalfCheetahWrapper
from environment.parametric_mujoco.parametric_pendulum_locomotion import MBPendulumAngleContinuousWrapper
from student_algorithms.gp_model_ensemble.dpgpmm_algorithm import DPGPMMAlgorithm
from student_algorithms.gp_model_ensemble.dpgpmm_policy import DPGPMMPolicy
from student_algorithms.nn_model_ensemble.nnmm_algorithm import NNMMAlgorithm
from student_algorithms.nn_model_ensemble.nnmm_policy import NNMMPolicy


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task):
    random_teacher = RandomTeacher(None, wrapper)

    os.environ["WANDB_MODE"] = "offline" # TODO: connection fix
    wandb.init(project='curriculum_rl', entity='liorf')
    config = wandb.config
    config.task = wrapper.name
    config.steps_per_task = steps_per_task
    config.num_tasks = tasks

    ref_env = wrapper.create_env(easy_task)

    student = NNMMAlgorithm(policy=NNMMPolicy, env=ref_env,
                            verbose=0,
                            env_state_reward_func=ref_env.reward_model(),
                            n_steps=1,
                            mm_burnin=20,
                            policy_kwargs={"net_arch": [8, 8]},
                            warm_up_time=800)  # Note: assumes all envs have a reward model

    config.student = str(student)
    config.student_params = student.__dict__

    wandb.watch(student.policy)  # TODO: put a model/policy (th module) and it logs gradients and model params

    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H') + "random small var leaking"
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)

    for i in tqdm(range(tasks)):
        random_teacher.train_k_actions(student, steps_per_task)
        wandb.log({"step": i,
                   "student_reward": random_teacher.history.history[-1][1],
                   "teacher_task": random_teacher.history.history[-1][0], })
        if i % 5 == 0 and i > 0:
            difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 10, 3, steps_per_task)
            wandb.log({"step": i,
                       "reward_std": difficulty_estimates.std(),
                       "mean_reward": difficulty_estimates.mean(),
                       "task_rewards": difficulty_estimates.mean(axis=1)})
            params_arr = np.array(task_params)
            param_names = [name for name in wrapper.parameters.keys()]
            for param_num in range(params_arr.shape[1]):
                wandb.log({"step": i,
                           f"task_reward_corrs_{param_names[param_num]}":
                               np.corrcoef(difficulty_estimates.mean(axis=1), params_arr[:, param_num])})
            with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_{i}.pkl", "wb") as fptr:
                pickle.dump((difficulty_estimates, task_params), fptr)
            # TODO: save does not work for nnmm/dpgpmm, missing attributes
            #student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student_{i}.agent")

    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(random_teacher.history.history, fptr)

    # eval and record video
    wandb.gym.monitor()  # Any env used with gym wrapper monitor will now be recorded
    evaluate(steps_per_task, random_teacher.generate_task()[0],
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


def run_cheetah():
    easy_params = {
        "expected_speed": 0.1,
    }
    measure_difficulty(1000, 250, HalfCheetahWrapper(), easy_params)


def run_ant():
    easy_params = {
        "goal_x": 0.1,
        "goal_y": 0.1
    }
    measure_difficulty(1000, 250, AntWrapper(), easy_params)


def run_pend():
    easy_params = {
        "goal_angle": 1.5,
    }
    measure_difficulty(1000, 250, MBPendulumAngleContinuousWrapper(), easy_params)


if __name__ == "__main__":
    run_cheetah()
    run_ant()
    run_pend()
