from curriculum.teacher import Teacher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from environment.environment_parameter import ContinuousParameter


def plot_reward_graph(teacher: Teacher, fname=None):
    history = teacher.history.history
    plt.scatter(range(len(history)), [x[1] for x in history])
    plt.xlabel('# tasks')
    plt.ylabel('mean reward')
    plt.show()
    if fname is not None:
        plt.savefig(fname)


def __has_similar_task(task, params, task_bins, continuous_sensativity):
    continuous_params = [k for k, v in params.items() if isinstance(v, ContinuousParameter)]
    for candidate in task_bins:
        is_similar = True
        for param_name in params.keys():
            if param_name in continuous_params:
                value_bin = continuous_sensativity * (params[param_name].max_val - params[param_name].min_val)
                if abs(candidate[param_name] - task[param_name]) > value_bin:  # too far apart
                    is_similar = False
                    break
            else:
                if candidate[param_name] != task[param_name]:  # not the same
                    is_similar = False
                    break
        if is_similar:
            return True
    return False


def plot_diversity_graph(teacher: Teacher, continuous_sensativity=0.05, fname=None):
    params = teacher.env_wrapper.parameters
    history = teacher.history.history
    task_bins = []
    unique_tasks_over_time = np.zeros(len(history))
    for i, (task, _) in enumerate(history):
        if not __has_similar_task(task, params, task_bins, continuous_sensativity):
            task_bins.append(task)
        unique_tasks_over_time[i] = len(task_bins)
    plt.scatter(range(len(unique_tasks_over_time)), unique_tasks_over_time)
    plt.xlabel('# tasks')
    plt.ylabel('# unique tasks')
    plt.show()
    if fname is not None:
        plt.savefig(fname)


def plot_tsne_task_distribution(teacher: Teacher, fname=None):
    params = teacher.env_wrapper.parameters
    history = teacher.history.history

    # 1 categorical
    tasks = np.zeros((len(history), len(params)))
    df = pd.DataFrame.from_records([x[0] for x in history])
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category').cat.codes
    embedder = TSNE(n_components=2,  # dimensions
                    perplexity=50.0)
    low_dim = embedder.fit_transform(df)
    rews = np.array([x[1] for x in history])
    normalized_rews = (rews - np.min(rews))/np.ptp(rews)
    plt.scatter(low_dim[:, 0], low_dim[:, 1], c=normalized_rews, cmap='Blues')

    plt.xlabel('task embedding X')
    plt.ylabel('task embedding Y')
    plt.colorbar()
    plt.show()
    if fname is not None:
        plt.savefig(fname)
    pass


def plot_eval_performance(teacher: Teacher, fname=None):
    eval_data = teacher.eval_data

    plt.scatter(range(len(eval_data)), [x["eval_reward"] for x in eval_data])
    plt.xlabel('# tasks')
    plt.ylabel('eval reward')
    plt.show()
    if fname is not None:
        plt.savefig(fname)


def plot_eval_to_pretrain_performance(teacher: Teacher, fname=None):
    eval_data = teacher.eval_data

    pretrain_postrain_diff = [x["eval_reward"] - x["pretrain_reward"] for x in eval_data]

    plt.scatter(range(len(pretrain_postrain_diff)), pretrain_postrain_diff)
    plt.xlabel('# tasks')
    plt.ylabel('eval post-pre reward')
    plt.show()
    if fname is not None:
        plt.savefig(fname)
