# learn param -> reward for each teacher, use to pick teacher
import random

from stable_baselines3.common.type_aliases import GymEnv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any


class SimpleNet(torch.nn.Module):

    def __init__(self, net_params, in_size, out_size):
        super(SimpleNet, self).__init__()
        self.fc_in = torch.nn.Linear(in_size, net_params[0])
        self.fc_out = torch.nn.Linear(net_params[-1], out_size)
        self.fc_layers = []
        for i, layer_size in enumerate(net_params):
            if i == len(net_params) - 1:
                pass
            else:
                layer = torch.nn.Linear(net_params[i], net_params[i + 1])
                self.fc_layers.append(layer)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for mid_layer in self.fc_layers:
            x = F.relu(mid_layer(x))
        x = self.fc_out(x)
        return x


class PredictionMixtureTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.teachers = teacher_parameters["teachers"]
        self.is_linear_regression = ("regression" in teacher_parameters) and teacher_parameters["regression"]
        self.known_teachers = []
        if self.is_linear_regression:
            self.teacher_predictors = [LinearRegression() for i in range(len(self.teachers))]

        else:
            net_params = teacher_parameters["network_dimensions"]
            self.teacher_predictors = [SimpleNet(net_params, len(environment_parameters.parameters), 1) for i in range(len(self.teachers))]
        self.chosen_teacher = 0

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        task_suggestions = [teacher.generate_task() for teacher in self.teachers]
        task_suggestion_params = [list(x[1].values())for x in task_suggestions]

        if len(self.known_teachers) < len(self.teachers):
            not_chosen_teachers = set(range(len(self.teachers))).difference(self.known_teachers)
            self.chosen_teacher = random.choice(list(not_chosen_teachers))
            self.known_teachers.append(self.chosen_teacher)
        else:
            if self.is_linear_regression:
                np_params = np.array(task_suggestion_params)
                predictions = np.array(
                    [self.teacher_predictors[i].predict(np_params[i, :].reshape(1, -1)) for i in range(len(self.teachers))])
                self.chosen_teacher = np.argmax(predictions)
            else:
                with torch.no_grad():
                    torch_params = [torch.tensor(x, dtype=torch.float32) for x in task_suggestion_params]
                    predictions = np.array([self.teacher_predictors[i](torch_params[i]).item() for i in range(len(self.teachers))])
                    self.chosen_teacher = np.argmax(predictions)

        return task_suggestions[self.chosen_teacher]

    def update_teacher_policy(self):
        self.teachers[self.chosen_teacher].history.history.append(self.history[-1])
        self.teachers[self.chosen_teacher].update_teacher_policy()

        if self.is_linear_regression:
            tasks = [x[0] for x in self.teachers[self.chosen_teacher].history.history]
            examples = pd.DataFrame.from_records(tasks)
            rewards = np.array([x[1] for x in self.teachers[self.chosen_teacher].history.history])
            self.teacher_predictors[self.chosen_teacher].fit(examples, rewards)
        else:
            teacher_net = self.teacher_predictors[self.chosen_teacher]
            params = torch.tensor(list(self.history[-1][0].values()), dtype=torch.float32)
            reward = torch.tensor([self.history[-1][1]], dtype=torch.float32)
            predict_loss = torch.nn.MSELoss()(reward, teacher_net(params))
            teacher_net.zero_grad()
            predict_loss.backward()
