import numpy as np


class History(object):
    def __init__(self, history_parameters=None):
        self.history = []
        # TODO: also have option for bounded length

    def update(self, task, trajectory, rewards, dones):
        reward = self.calculate_reward(dones, rewards)  # TODO: discount
        self.history.append((task, reward))

    def calculate_reward(self, dones, rewards):
        rews = []
        rew = 0
        for i in range(len(rewards)):
            rew += rewards[i] * (1 - dones[i])
            if dones[i] == 1:
                rews.append(rew)
                rew = 0
        if len(rews) == 0:
            return rew
        else:  # ignore last iteration, we can't use it
            return np.mean(rews)
