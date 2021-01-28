# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
from typing import Dict, Any

import gym
import procgen
from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.eval.history_metrics import plot_reward_graph, plot_diversity_graph, plot_tsne_task_distribution, \
    plot_eval_performance, plot_eval_to_pretrain_performance
from curriculum.teacher import Teacher
from curriculum.teachers.adr_teacher import AdrTeacher
from curriculum.teachers.again_teacher import AgainTeacher
from curriculum.teachers.alp_gmm_teacher import AlpGmmTeacher
from curriculum.teachers.learning_rate_sampling_teacher import LearningRateSamplingTeacher
from curriculum.teachers.learning_rate_teacher import LearningRateTeacher
from curriculum.teachers.mixture_teachers.const_change_mixture_teacher import ConstMixtureTeacher
from curriculum.teachers.mixture_teachers.predicted_value_mixture_teacher import PredictionMixtureTeacher
from curriculum.teachers.mixture_teachers.reward_stepsize_mixture_teacher import RewardStepsizeMixtureTeacher
from curriculum.teachers.mixture_teachers.reward_weighted_mixture_teacher import RewardMixtureTeacher
from curriculum.teachers.random_teacher import RandomTeacher
from curriculum.teachers.reward_shaping.learned_q_shaping import LearnedQShaping
from curriculum.teachers.reward_shaping.learned_v_shaping import LearnedVShaping
from curriculum.teachers.reward_shaping.long_episode_shaping import LongEpisodeShaping
from curriculum.teachers.reward_shaping.new_state_shaping import NewStateShaping
from curriculum.teachers.riac_teacher import RiacTeacher
from environment.environment_wrapper import EnvironmentWrapper
from environment.parametric_walker_env.bodies.BodyTypesEnum import BodyTypesEnum
from environment.parametric_walker_env.parametric_continuous_flat_parkour import ParametricContinuousWalker
from environment.parametric_walker_env.parametric_walker_wrapper import get_classic_walker
from environment.simple_envs.parametric_cartpole import CartpoleWrapper


def learn_parameteric():
    env = get_classic_walker().create_env({
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
        "motors_torque": 80
    })
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=1024)
    student.learn(20000)

    eval_run(student, env)


def eval_run(student, env):
    s = env.reset()

    for i in range(200):
        a, _ = student.predict(observation=s)
        env.render()
        s, r, d, _ = env.step(a)
        if d == 1:
            print(r)
            env.close()
            return
    env.close()
    print(r)


def check_sanity():
    env_id = "CartPole-v1"
    student = PPO(policy='MlpPolicy', env=env_id, batch_size=100, verbose=0, n_steps=200)

    # student = A2C(policy='MlpPolicy', env=env_id, verbose=0, n_steps=200)

    # student = DQN(policy='MlpPolicy', env=env_id, verbose=0, learning_starts=0, buffer_size=10000)

    # teacher = RandomTeacher(None, CartpoleWrapper())

    teacher = LearningRateTeacher({"alpha": 0.9, "epsilon": 0.1}, CartpoleWrapper())

    # teacher = LearningRateSamplingTeacher({"k": 5}, CartpoleWrapper())

    for i in range(200):
        teacher.train_k_actions(student, 200)
    # plot_reward_graph(teacher)
    # plot_diversity_graph(teacher)
    # plot_tsne_task_distribution(teacher)


def check_continuous(eval=False):
    env_params = {
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
        # "motors_torque": 80
    }
    env = get_classic_walker().create_env(env_params)
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=200)

    # teacher = RiacTeacher({"max_region_size": 30}, get_classic_walker())

    # teacher = AdrTeacher({"reward_thr": 0, "initial_task": [0, 10, 3, 6]}, get_classic_walker())

    # teacher = AlpGmmTeacher({"gmm_fitness_fun": "aic", "fit_rate": 20}, get_classic_walker())

    teacher = AgainTeacher({"gmm_fitness_fun": "aic", "fit_rate": 20, "student_params": {}}, get_classic_walker())

    for i in range(100):
        if eval:
            teacher.train_k_actions(student, 400, eval_task_params=env_params, pretrain=True)
        else:
            teacher.train_k_actions(student, 400)

    plot_reward_graph(teacher)
    plot_diversity_graph(teacher)
    plot_tsne_task_distribution(teacher)
    if eval:
        plot_eval_performance(teacher)
        plot_eval_to_pretrain_performance(teacher)
    eval_run(student, env)


def check_mixture(eval=False):
    env_params = {
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
    }
    env = get_classic_walker().create_env(env_params)
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=200)

    teacher1 = RandomTeacher(None, get_classic_walker())
    teacher2 = RiacTeacher({"max_region_size": 30}, get_classic_walker())
    teacher3 = AdrTeacher({"reward_thr": 0, "initial_task": [0, 10, 3, 6]}, get_classic_walker())

    # teacher = PredictionMixtureTeacher({"teachers": [teacher1, teacher2, teacher3],
    #                                     "regression": True}, get_classic_walker())

    teacher = PredictionMixtureTeacher({"teachers": [teacher1, teacher2, teacher3],
                                        "network_dimensions": [16, 16]}, get_classic_walker())

    # teacher = ConstMixtureTeacher({"teachers": [teacher1, teacher2, teacher3],
    #                                 "step_size": 0.1}, get_classic_walker())

    # teacher = RewardStepsizeMixtureTeacher({"teachers": [teacher1, teacher2, teacher3],
    #                                 "step_size": 0.1}, get_classic_walker())

    # teacher = RewardMixtureTeacher({"teachers": [teacher1, teacher2, teacher3],
    #                                 "step_size": 0.1}, get_classic_walker())

    for i in range(100):
        if eval:
            teacher.train_k_actions(student, 400, eval_task_params=env_params, pretrain=True)
        else:
            teacher.train_k_actions(student, 400)

    plot_reward_graph(teacher)
    plot_diversity_graph(teacher)
    plot_tsne_task_distribution(teacher)
    if eval:
        plot_eval_performance(teacher)
        plot_eval_to_pretrain_performance(teacher)
    eval_run(student, env)


def check_shaping(eval=False):
    env_params = {
        "climbing_surface_size": 0,
        "gap_size": 10,
        "gap_pos": 3,
        "obstacle_spacing": 6,
    }
    env = get_classic_walker().create_env(env_params)
    student = PPO(policy='MlpPolicy', env=env, verbose=0, n_steps=200)

    teacher1 = RandomTeacher(None, get_classic_walker())

    # shaped_teacher = LearnedVShaping({"scale": 1.0,
    #                                   "step_size": 0.1,
    #                                   "obs_shape": env.observation_space.shape[0],
    #                                   "network_dimensions": [16, 16],
    #                                   "discount": 0.9},
    #                                  get_classic_walker(), teacher1)

    shaped_teacher = LearnedQShaping({"scale": 1.0,
                                      "step_size": 0.1,
                                      "obs_shape": env.observation_space.shape[0],
                                      "action_shape": env.action_space.shape[0],
                                      "network_dimensions": [16, 16],
                                      "discount": 0.9},
                                     get_classic_walker(), teacher1)

    # shaped_teacher = LongEpisodeShaping({"scale": 1.0,
    #                                      "is_strong": False},
    #                                     get_classic_walker(), teacher1)

    # shaped_teacher = NewStateShaping({"scale": 1.0,
    #                                   "state_distance": 0.5,
    #                                   "new_state_reward": 1.0},
    #                                  get_classic_walker(), teacher1)

    # shaped_teacher = KnownStateShaping({"scale": 1.0,
    #                                   "state_distance": 0.5,
    #                                   "familiar_state_reward": 1.0},
    #                                  get_classic_walker(), teacher1)

    for i in range(100):
        if eval:
            shaped_teacher.train_k_actions(student, 400, eval_task_params=env_params, pretrain=True)
        else:
            shaped_teacher.train_k_actions(student, 400)

    plot_reward_graph(shaped_teacher)
    plot_diversity_graph(shaped_teacher)
    plot_tsne_task_distribution(shaped_teacher)
    if eval:
        plot_eval_performance(shaped_teacher)
        plot_eval_to_pretrain_performance(shaped_teacher)
    eval_run(student, env)


# check_sanity()
# check_continuous(eval=False)
# check_mixture(eval=False)
check_shaping(eval=False)
