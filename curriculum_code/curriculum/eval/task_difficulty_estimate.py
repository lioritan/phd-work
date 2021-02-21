from stable_baselines3 import A2C

from curriculum.teacher import Teacher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.utils.task_space_utils import has_similar_task
from environment.environment_parameter import ContinuousParameter, DiscreteParameter
from environment.environment_wrapper import EnvironmentWrapper
from environment.simple_envs.parametric_cartpole import CartpoleWrapper


def estimate_task_difficulties(student, task_wrapper: EnvironmentWrapper, num_segments=5, trials_per_task=3,
                               max_episode_len=200):
    task_estimates = np.zeros((num_segments ** len(task_wrapper.parameters), trials_per_task))
    task_params = []
    free_ind = 0

    dummy_teacher = RandomTeacher(None, task_wrapper)

    param_names = list(task_wrapper.parameters.keys())
    param_ranges = [np.arange(task_wrapper.parameters[n].min_val,
                              task_wrapper.parameters[n].max_val+1,
                              (task_wrapper.parameters[n].max_val - task_wrapper.parameters[n].min_val) / (num_segments-1))
                    for n in param_names]

    indices = np.zeros(len(param_names), dtype=np.int)
    while free_ind < task_estimates.shape[0]:
        chosen_params = [param_ranges[i][indices[i]] for i in range(len(param_names))]
        for i, param_name in enumerate(param_names):
            if isinstance(task_wrapper.parameters[param_name], DiscreteParameter):
                chosen_params[i] = int(round(chosen_params[i]))
        task_params.append(chosen_params)
        for trial_num in range(trials_per_task):
            try:
                total_reward, _ = dummy_teacher.evaluate(max_episode_len,
                                                     {param_names[i]: chosen_params[i] for i in range(len(param_names))},
                                                         student)
            except:
                print(chosen_params)
                total_reward = 0
            task_estimates[free_ind, trial_num] = total_reward

        free_ind += 1
        indices[-1] = (indices[-1] + 1) % num_segments
        if indices[-1] == 0:
            for i in range(-2, -len(indices)-1, -1):
                indices[i] = (indices[i] + 1) % num_segments
                if indices[i] != 0:
                    break

    return task_estimates, task_params


def plot_estimated_task_difficulties(task_estimates, task_params, filename=None):

    avg_task_estimates = task_estimates.mean(axis=1)
    tasks = pd.DataFrame(task_params)

    embedder = TSNE(n_components=2,  # dimensions
                     perplexity=50.0)
    low_dim = embedder.fit_transform(tasks)
    normalized_rews = (avg_task_estimates - np.min(avg_task_estimates)) / np.ptp(avg_task_estimates)
    plt.scatter(low_dim[:, 0], low_dim[:, 1], c=normalized_rews, cmap='Blues')

    plt.xlabel('task embedding X')
    plt.ylabel('task embedding Y')
    plt.colorbar()
    if filename:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()
