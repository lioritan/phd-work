import gym
import torch as th
from typing import Callable, Optional, List, Type, Dict, Any
from torch import nn
import numpy as np

from student_algorithms.nn_model_ensemble.nnmm.single_nn import DynamicsNetwork


class NNMixture(nn.Module):
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
        super(NNMixture, self).__init__()
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
        self.assigns = []
        self.rho_sum = []
        self.new_net = None
        self.new_net_points = 0

    def init_net(self):
        return DynamicsNetwork(self.obs_space, self.action_space, self.lr_schedule,
                               self.net_arch, self.activation_fn, self.optimizer_class,
                               self.optimizer_kwargs)

    def add_net(self, new_net):
        self.networks.append(new_net)
        self.n_nets += 1

    def add_example(self, data):
        self.data.append(data)

    def fit(self):
        # get the newest data point x_{n+1}
        i = len(self.data) - 1
        x = self.data[i]

        # calculate \rho_{n+1, 1:k} - [k, 1]
        # TODO: replace log_posterior_pdf, tensorify
        rho_old = [self.rho_sum[k] * np.exp(self.comps[k].log_posterior_pdf(x)) for k in range(self.n_nets)]

        # create a new component and use the new data point as training data
        if self.new_net is None:
            self.new_net = self.init_net()
            self.new_net_points = 0

        pass # TODO: do one step for net on x

        # calculate \rho_{n+1, k+1} - [1, 1]
        if self.n_nets > 0 and self.new_net_points < self.merge_burnin:
            rho_new = [0.0]  # can afford to wait on testing the new net
        else:
            # TODO
            rho_new = [self.alpha * np.exp(self.new_net.log_posterior_pdf(x))]

        # normalize \rho
        rho = rho_old + rho_new
        rho = rho / np.sum(rho)

        # get the max probability index
        k = np.argmax(rho, axis=0)
        self.assigns.append(k)
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
            pass # TODO: add point to network k and train it
        else:
            pass
            # logger.error('Index exceeds the length of components!')

        # TODO: merge

        return self.alpha

    def forward(self, obs: th.Tensor, action: th.Tensor):
        most_likely_net = self.assigns[-1]
        with th.no_grad():
            return self.networks[most_likely_net](obs, action)
