import gym
import torch as th
from typing import Callable, Optional, List, Type, Dict, Any
from torch import nn
import numpy as np
import torch.nn.functional as F

from student_algorithms.nn_model_ensemble.nnmm.single_nn import DynamicsNetwork


class NNMixtureWeighted(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 alpha: float,
                 window_prob: float,  # initial bias for new model to transition to other states
                 self_prob: float,  # initial bias for new model to transition to itself
                 merge_burnin: int,  # number of points until we consider merge
                 merge_threshold: float,  # min KL-div for merge

                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, ):
        super(NNMixtureWeighted, self).__init__()
        self.obs_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.networks = []
        self.n_nets = 0

        self.alpha = alpha
        self.window_prod = window_prob
        self.self_prod = self_prob
        self.merge_burnin = merge_burnin
        self.merge_thresh = merge_threshold

        self.data = []
        self.assign_probs = []
        self.rho_sum = []
        self.new_net = None

    def init_net(self):
        return DynamicsNetwork(self.obs_space, self.action_space, self.lr_schedule,
                               self.net_arch, self.activation_fn, self.optimizer_class,
                               self.optimizer_kwargs)

    def add_net(self, new_net):
        self.networks.append(new_net)
        self.n_nets += 1

    def add_example(self, data):
        self.data.append(data)

    def do_net_step(self, net, x):
        with th.no_grad():
            predicted_data = net(x[0], x[1])
        loss = F.mse_loss(predicted_data, x[2])
        net.optimizer.zero_grad()
        loss.backward()
        net.optimizer.step()
        net.n_points += 1

    def fit(self):
        # get the newest data point x_{n+1}
        i = len(self.data) - 1
        x = self.data[i]

        # calculate \rho_{n+1, 1:k} - [k, 1]
        rho_old = [self.rho_sum[k] * np.exp(self.comps[k].log_likelihood(x[0], x[1], x[2]))
                   for k in range(self.n_nets)]

        # create a new component and use the new data point as training data
        if self.new_net is None:
            self.new_net = self.init_net()

        self.do_net_step(self.new_net, x)

        # calculate \rho_{n+1, k+1} - [1, 1]
        if self.n_nets > 0 and self.new_net.n_points < self.merge_burnin:
            rho_new = [0.0]  # can afford to wait on testing the new net
        else:
            rho_new = [self.alpha * np.exp(self.new_net.log_likelihood(x[0], x[1], x[2]))]

        # normalize \rho
        rho = rho_old + rho_new
        rho = rho / np.sum(rho)

        # get the max probability index
        k = np.argmax(rho, axis=0)
        self.assign_probs.append(th.tensor(rho))
        # TODO: assign the point to all models by weight
        if k == self.n_nets:
            # \rho_{1:k+1} = [\rho_{1:k}, \rho_{k+1}]
            self.rho_sum.extend([0])
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.add_net(self.new_net)
            self.new_net = None
        elif k < self.n_nets:
            rho_old = rho_old / np.sum(rho_old)
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho_old)]
            self.do_net_step(self.networks[k], x)
        else:
            pass
            # logger.error('Index exceeds the length of components!')

        # TODO: merge / another way to remove the new net

        return self.alpha

    def forward(self, obs: th.Tensor, action: th.Tensor):
        # keep last "likelihood distrib"
        # do weighted average
        net_probs = self.assign_probs[-1]
        with th.no_grad():
            return th.sum(net_probs * th.tensor([net(obs, action) for net in self.networks]))
