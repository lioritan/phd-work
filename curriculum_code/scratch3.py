from stable_baselines3 import DQN, PPO, A2C

from curriculum.teachers.predefined_tasks_teacher import PredefinedTasksTeacher
from curriculum.teachers.reward_shaping.reward_shaping_wrapper_teacher import RewardShapingTeacher
from environment.gridworld_advanced.parametric_gridworld_advanced import GridworldsCustomWrapper
from student_algorithms.ensemble_dqn.ebql import EBQL
import numpy as np

class GoalShaping(RewardShapingTeacher):

    def __init__(self, teacher_parameters, environment_parameters, base_teacher):
        super().__init__(teacher_parameters, environment_parameters, base_teacher)
        self.is_goal = False
        self.last_s = None

    def shaping_function(self, s, a):
        if a > 2:
            return -100
        if self.is_goal:
            return 1
        elif (s==8).any():
            loc = [x for x in np.where(s==8)]
            loc[1] = abs(loc[1]-2)
            return 1 - sum(loc)[0]/20
        else:
            return -100

    def shaping_step_update_function(self, s, a, r, s_new, done):
        super(GoalShaping, self).shaping_step_update_function(s, a, r, s_new, done)
        if done==1:
            self.is_goal = True

    def update_shaping_function(self):
        self.is_goal = False

def eval_run(student, env):
    s = env.reset()

    for i in range(110):
        a, _ = student.predict(observation=s, deterministic=True)
        env.render()
        s, r, d, _ = env.step(a)
        if d == 1:
            print(r)
            env.close()
            return
    env.close()
    print(r)


easy_params = {
        "depth": 2,
        "width": 1,
        "keys": 0,
        "maze_percentage": 0.0,
    }
ref_env = GridworldsCustomWrapper().create_env(easy_params)

teacher = PredefinedTasksTeacher({"tasks": [easy_params]}, GridworldsCustomWrapper())
shaped_teacher = GoalShaping({"scale": 1}, {}, teacher)
#shaped_teacher = teacher

student = EBQL(policy='MlpPolicy', env=ref_env, learning_starts=10, tau=1.0, exploration_fraction=1.0, exploration_initial_eps=1.0, exploration_final_eps=0.4,
             ensemble_size=3, verbose=0, policy_kwargs={"net_arch": [32]}, buffer_size=10000, learning_rate=1e-3
             )

student = DQN(policy='MlpPolicy', env=ref_env, learning_starts=10, tau=1.0, exploration_fraction=1.0, exploration_initial_eps=1.0, exploration_final_eps=0.4,
              verbose=0, policy_kwargs={"net_arch": [32]}, buffer_size=10000, learning_rate=1e-3
             )

student = A2C(policy='MlpPolicy', env=ref_env,
              verbose=0, policy_kwargs={"net_arch": [dict(pi=[8,8], vf=[8,8])]}, learning_rate=1e-3,
             )

for i in range(100):
    shaped_teacher.train_k_actions(student, 200)
eval_run(student, ref_env)

print("hello")
