# TODO: parse teacher params, parse env params, run & evaluate
# consider splitting some of the code here
import gym
from stable_baselines3.common.utils import constant_fn
import stable_baselines3.dqn as dqn

from curriculum.students.dqn import DQN
from curriculum.teacher import Teacher


class SanTeacher(Teacher):

    def update_teacher_policy(self):
        pass

    def generate_task(self):
        return gym.make('CartPole-v1')


def check_sanity():
    #student = PPO(policy='MlpPolicy', env="CartPole-v1", batch_size=1)
    env = gym.make("CartPole-v1")
    policy = dqn.MlpPolicy(observation_space=env.observation_space, action_space=env.action_space,
                           lr_schedule=constant_fn(0.1))
    policy = policy.to("cuda")
    student = DQN(policy, env, learning_rate=0.1, learning_starts=10, gradient_steps=1, tau=0.8)

    teacher = SanTeacher(None, None)
    for i in range(101):
        teacher.train_single_episode(student, 200)

check_sanity()
