import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from curriculum.teachers.random_teacher import RandomTeacher
from environment.simple_envs.parametric_lunarlander import LunarLanderWrapper
from environment.simple_envs.parametric_lunarlander_continuous import LunarLanderContinuousWrapper
from student_algorithms.pets.pets import PETS
from task_difficulty_experiment import ParamLeakingEnvWrapper

# with open(r"D:\GitHub\phd-work\curriculum_code\results\2021-03-15\difficulty\LunarLanderContinuous-v2\data.pkl",
#           "rb") as fptr:
#     estimates, params = pickle.load(fptr)
#
# avg_task_estimates = estimates.mean(axis=1)
# tasks = pd.DataFrame(params)
wrapper = LunarLanderWrapper()
# random_teacher = RandomTeacher(None, ParamLeakingEnvWrapper(LunarLanderWrapper()))
random_teacher = RandomTeacher(None, wrapper)
# with open(r"D:\GitHub\phd-work\curriculum_code\results\2021-03-15\difficulty\LunarLanderContinuous-v2\hist.pkl",
#           "rb") as fptr:
#     hist = pickle.load(fptr)
# random_teacher.history = hist

# student = PETS.load(
#     r"D:\GitHub\phd-work\curriculum_code\results\2021-03-15\difficulty\LunarLanderContinuous-v2\student.agent",
#     env=random_teacher.generate_task()[0])

student = PPO.load(r"D:\GitHub\phd-work\curriculum_code\results\2021-04-06 11\difficulty\LunarLander-v2\student.agent", env=random_teacher.generate_task()[0])

def eval_run(student, env):
    s = env.reset()

    for i in range(200):
        a, _ = student.predict(observation=s.reshape(1, -1))
        env.render()
        s, r, d, _ = env.step(a.reshape(-1))
        if d == 1:
            print(r)
            env.close()
            return
    env.close()
    print(r)


#eval_run(student, random_teacher.generate_task()[0])
# TODO: fails on load->policy init-> _init_setup_model
estimated_hist_vals = np.zeros((1000, 3))


def evaluate(action_limit, eval_task_params, env_wrapper, student):
    eval_env = env_wrapper.create_env(eval_task_params)
    student.set_env(eval_env)
    s = eval_env.reset()
    eval_env.render()
    total_reward = 0
    episode_length = 0
    for i in range(action_limit):
        a, _ = student.predict(observation=s.reshape(1, -1), deterministic=True)
        s, r, d, _ = eval_env.step(a[0])
        eval_env.render()
        episode_length += 1
        total_reward += r
        if d == 1:
            break
    eval_env.close()
    return total_reward, episode_length


evaluate(400, {"leg_height": 20,
        "leg_width": 2,
        "main_engine_power": 10.0,
        "side_engine_power": 8.0,}, wrapper, student)
exit()

hist_tasks = pd.DataFrame.from_records([x[0] for x in random_teacher.history])
actual_val = np.array([x[1] for x in random_teacher.history])
for t, task in enumerate([x[0] for x in random_teacher.history]):
    if t == 1000:
        break
    for i in [0, 1, 2]:
        r, _ = evaluate(400, task, wrapper, student)
        estimated_hist_vals[t, i] = r

print(np.corrcoef(estimated_hist_vals.mean(axis=1), hist_tasks))

plt.scatter(hist_tasks["leg_height"], estimated_hist_vals.mean(axis=1))
plt.scatter(hist_tasks["leg_height"], estimated_hist_vals.mean(axis=1) - actual_val)
plt.scatter(range(1000), estimated_hist_vals.mean(axis=1) - actual_val)

plt.show()

print(666)
