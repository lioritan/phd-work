# consider splitting some of the code here
import datetime
import os
import pickle

from tqdm import tqdm

from curriculum.eval.task_difficulty_estimate import estimate_task_difficulties
from curriculum.teachers.random_teacher import RandomTeacher
from environment.parametric_mujoco.parametric_ant import AntWrapper
from environment.parametric_mujoco.parametric_half_cheetah import HalfCheetahWrapper
from student_algorithms.gp_model_ensemble.dpgpmm_algorithm import DPGPMMAlgorithm
from student_algorithms.gp_model_ensemble.dpgpmm_policy import DPGPMMPolicy


def measure_difficulty(steps_per_task, tasks, wrapper, easy_task):
    random_teacher = RandomTeacher(None, wrapper)

    # wandb.init(project='curriculum_rl', entity='liorf')
    # config = wandb.config # TODO: experiemnt config
    # wandb.watch(None) # TODO: put a model/policy (th module) and it logs gradients and model params
    # wandb.log({"loss": 6}) # TODO: metric
    # wandb.gym.monitor() # makes video of env

    ref_env = wrapper.create_env(easy_task)

    student = DPGPMMAlgorithm(policy=DPGPMMPolicy, env=ref_env,
                              verbose=0,
                              env_state_reward_func=ref_env.reward_model(),
                              n_steps=1,
                              warm_up_time=500)  # Note: assumes all envs have a reward model

    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H') + "random small var leaking"
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)

    for i in tqdm(range(tasks)):
        random_teacher.train_k_actions(student, steps_per_task)
        if i % 50 == 0 and i > 0:
            difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 10, 3, steps_per_task)
            with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_{i}.pkl", "wb") as fptr:
                pickle.dump((difficulty_estimates, task_params), fptr)
            student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student_{i}.agent")

    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(random_teacher.history.history, fptr)


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


if __name__ == "__main__":
    run_cheetah()
    run_ant()
