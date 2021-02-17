from abc import ABC, abstractmethod
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import get_obs_shape, get_action_dim
from stable_baselines3.common.type_aliases import GymEnv

from curriculum.history import History
from typing import Dict, Tuple, Any

from environment.environment_parameter import EnvironmentParameter


class StudentTrackingCallback(BaseCallback):
    def __init__(self, num_actions, obs_space, action_space):
        super(StudentTrackingCallback, self).__init__(verbose=0)
        self.obs = np.zeros((num_actions, ) + get_obs_shape(obs_space), dtype=np.float32)
        self.action = np.zeros((num_actions, get_action_dim(action_space)), dtype=np.float32)
        self.reward = np.zeros(num_actions, dtype=np.float32)
        self.done = np.zeros(num_actions, dtype=np.float32)
        self.ind = 0
        self.num_actions = num_actions
        self.wtf_count = 0

    def _on_step(self) -> bool:
        if self.ind == self.num_actions:
            self.wtf_count += 1
            return True
        done_array = np.array(
            self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        #self.obs[self.ind, :] = self.locals.get("obs_tensor").cpu().numpy() if self.locals.get("obs_tensor") is not None else self.locals.get("self")._last_obs
        #self.action[self.ind, :] = self.locals.get("actions") if self.locals.get("actions") is not None else self.locals.get("buffer_action")
        self.reward[self.ind] = self.locals.get("rewards") if self.locals.get("rewards") is not None else self.locals.get("reward")
        self.done[self.ind] = done_array
        self.ind += 1

        return True


class Teacher(ABC):
    def __init__(self, teacher_parameters, environment_wrapper, history_parameters=None, seed=None):
        self.history = History(history_parameters)
        self.env_wrapper = environment_wrapper
        self.seed = seed
        self.eval_data = []

    def train_k_actions(self, student: BaseAlgorithm, action_limit: int, eval_task_params=None, pretrain=False):
        training_task, params = self.generate_task()
        trajectory, rewards, dones = self.train_student_on_task(student, training_task, action_limit, eval_task_params, pretrain)
        self.history.update(params, trajectory, rewards, dones)
        self.update_teacher_policy()

    def train_student_on_task(self, student, training_task, action_limit, eval_task_params=None, pretrain=False):
        eval_dict = {}
        if pretrain and eval_task_params is not None:
            total_reward, episode_length = self.evaluate(action_limit, eval_task_params, student)
            eval_dict["pretrain_reward"] = total_reward

        student.set_env(training_task)
        data_callback = StudentTrackingCallback(action_limit, training_task.observation_space, training_task.action_space)
        student.learn(action_limit, callback=data_callback)

        if eval_task_params is not None:
            total_reward, episode_length = self.evaluate(action_limit, eval_task_params, student)
            eval_dict["eval_reward"] = total_reward

        if len(eval_dict) > 0:
            self.eval_data.append(eval_dict)

        # return the trajectory and rewards
        return (data_callback.obs, data_callback.action), data_callback.reward, data_callback.done

    def evaluate(self, action_limit, eval_task_params, student):
        eval_env = self.env_wrapper.create_env(eval_task_params)
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

    @abstractmethod
    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        pass

    @abstractmethod
    def update_teacher_policy(self):
        pass
