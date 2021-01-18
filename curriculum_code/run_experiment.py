# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
import gym
import procgen
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.teacher import Teacher
from curriculum.teachers.random_env_teacher import RandTeacher


class SanTeacher(Teacher):

    def update_teacher_policy(self):
        pass

    def generate_task(self):
        return gym.make('CartPole-v1')

def check_sanity():
    env_id = "CartPole-v1"
    student = PPO(policy='MlpPolicy', env=env_id, batch_size=100, verbose=0, n_steps=200)

    #student = A2C(policy='MlpPolicy', env=env_id, verbose=1, n_steps=200)

    #student = DQN(policy='MlpPolicy', env=env_id, verbose=0, learning_starts=0, buffer_size=10000)
    env = gym.make(env_id)

    teacher = SanTeacher(None, None)
    for i in range(101):
        teacher.train_k_actions(student, 200)

check_sanity()
