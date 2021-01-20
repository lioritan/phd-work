from collections import deque

from gym.spaces import Box
from stable_baselines3.common.type_aliases import GymEnv

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any
import numpy as np


# Automatic Domain Randomization, see https://arxiv.org/abs/1910.07113 for details
class AdrTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        # Task space boundaries
        self.ordered_params = list(environment_parameters.parameters.keys())
        self.mins = np.array([environment_parameters.parameters[p_name].min_val for p_name in self.ordered_params])
        self.maxs = np.array([environment_parameters.parameters[p_name].max_val for p_name in self.ordered_params])
        self.nb_dims = len(self.ordered_params)

        # initial calibrating task
        self.initial_task = np.array(
            [0] * len(self.mins) if "initial_task" not in teacher_parameters else teacher_parameters["initial_task"])
        print('initial task is {}'.format(self.initial_task))

        # Boundary sampling probability p_r
        self.bound_sampling_p = 0.5 if "boundary_sampling_p" not in teacher_parameters else teacher_parameters[
            "boundary_sampling_p"]

        # ADR step size
        self.step_size = [0.02] * self.nb_dims if "step_size" not in teacher_parameters else teacher_parameters[
            "step_size"]

        # Increase threshold
        self.reward_threshold = 230 if "reward_thr" not in teacher_parameters else teacher_parameters["reward_thr"]
        self.reward_threshold = np.interp(self.reward_threshold, (-150, 350), (0, 1))
        # max queue length
        self.window_len = 10 if "queue_len" not in teacher_parameters else teacher_parameters['queue_len']
        hyperparams = locals()
        # Set initial task space to predefined calibrated task
        self.cur_mins = np.array(self.initial_task, dtype=np.float32)  # current min bounds
        self.cur_maxs = np.array(self.initial_task, dtype=np.float32)  # current max bounds
        self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)

        # Init queues, one per task space dimension
        self.min_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]
        self.max_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]

        # Boring book-keeping
        self.episode_nb = 0
        print(hyperparams)
        self.bk = {'task_space': [(self.cur_mins.copy(), self.cur_maxs.copy())],
                   'episodes': [],
                   'ADR_hyperparams': hyperparams}

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        new_task = self.task_space.sample()
        if np.random.random() < self.bound_sampling_p:  # set random dimension to min or max bound
            idx = np.random.randint(0, self.nb_dims)
            is_min_max_capped = np.array([self.cur_mins[idx] == self.mins[idx], self.cur_maxs[idx] == self.maxs[idx]])
            if not is_min_max_capped.all():  # both min and max bounds can increase, choose extremum randomly
                if np.random.random() < 0.5:  # skip min bound if already
                    new_task[idx] = self.cur_mins[idx]
                else:
                    new_task[idx] = self.cur_maxs[idx]
            elif not is_min_max_capped[0]:
                new_task[idx] = self.cur_mins[idx]
            elif not is_min_max_capped[1]:
                new_task[idx] = self.cur_maxs[idx]

        task_params = self.box_to_params(new_task)
        return self.env_wrapper.create_env(task_params), task_params

    def update_teacher_policy(self):
        task_params, reward = self.history[-1]
        task = self.params_to_array(task_params)
        self.episode_nb += 1
        # check for updates
        for i, (min_q, max_q, cur_min, cur_max) in enumerate(
                zip(self.min_queues, self.max_queues, self.cur_mins, self.cur_maxs)):
            if task[i] == cur_min:  # if the proposed task has the i^th dimension set to min boundary
                min_q.append(reward)
                if len(min_q) == self.window_len and np.mean(min_q) >= self.reward_threshold:  # decrease boundary
                    self.cur_mins[i] = max(self.cur_mins[i] - self.step_size[i], self.mins[i])
                    self.min_queues[i] = deque(maxlen=self.window_len)  # reset queue
            if task[i] == cur_max:  # if the proposed task has the i^th dimension set to max boundary
                max_q.append(reward)
                if len(max_q) == self.window_len and np.mean(max_q) >= self.reward_threshold:  # decrease boundary
                    self.cur_maxs[i] = min(self.cur_maxs[i] + self.step_size[i], self.maxs[i])
                    self.max_queues[i] = deque(maxlen=self.window_len)  # reset queue

        prev_cur_mins, prev_cur_maxs = self.bk['task_space'][-1]
        # print(self.bk['task_space'][-1])
        if (prev_cur_mins != self.cur_mins).any() or (
                prev_cur_maxs != self.cur_maxs).any():  # were boundaries changed ?
            self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)
            # book-keeping only if boundaries were updates
            self.bk['task_space'].append((self.cur_mins.copy(), self.cur_maxs.copy()))
            self.bk['episodes'].append(self.episode_nb)
            print(self.bk['task_space'][-1])

    def box_to_params(self, box_sample):
        return {self.ordered_params[i]: box_sample[i] for i in range(len(self.ordered_params))}

    def params_to_array(self, params):
        return np.array([params[p] for p in self.ordered_params])
