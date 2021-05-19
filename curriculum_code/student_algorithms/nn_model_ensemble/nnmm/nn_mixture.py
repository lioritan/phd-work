import gym
import torch as th
from typing import Callable, Optional, List, Type, Dict, Any
from torch import nn
import numpy as np
import torch.nn.functional as F
from student_algorithms.nn_model_ensemble.nnmm.single_nn import DynamicsNetwork
import wandb


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
                 n_epochs: int,
                 is_res_net: bool,

                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, ):
        super(NNMixture, self).__init__()
        self.obs_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.is_res_net = is_res_net
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
        self.n_epochs = n_epochs

        self.data = []
        self.assigns = []
        self.rho_sum = []
        self.new_net = None

        self.reset_priors()

    def init_net(self):
        net = DynamicsNetwork(self.obs_space, self.action_space, self.lr_schedule, self.is_res_net,
                              self.net_arch, self.activation_fn, self.optimizer_class,
                              self.optimizer_kwargs)
        net.to(self.device)
        return net

    def add_net(self, new_net):
        self.networks.append(new_net)
        self.n_nets += 1

    def add_example(self, data):
        self.data.append(data)

    def do_net_step(self, net, x, log_losses=False):
        for i in range(self.n_epochs):
            predicted_data = net(x[0], x[1])
            loss = F.mse_loss(predicted_data, x[2])
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            if log_losses:
                wandb.log({
                    "loss": loss.cpu().item()
                })
        net.n_points += 1

    def fit(self):
        # get the newest data point x_{n+1}
        i = len(self.data) - 1
        x = self.data[i]
        # TODO: make all of this code torch so it's faster

        # calculate \rho_{n+1, 1:k} - [k, 1]
        rho_old = [self.rho_sum[k] * th.exp(self.networks[k].log_likelihood(x[0], x[1], x[2]))
                   for k in range(self.n_nets)]
        rho_old = [rho_k.item() for rho_k in rho_old]

        # create a new component and use the new data point as training data
        if self.new_net is None:
            self.new_net = self.init_net()

        self.do_net_step(self.new_net, x)

        # calculate \rho_{n+1, k+1} - [1, 1]
        if self.n_nets > 0 and self.new_net.n_points < self.merge_burnin:
            rho_new = [0.0]  # can afford to wait on testing the new net
        else:
            rho_new = [self.alpha * th.exp(self.new_net.log_likelihood(x[0], x[1], x[2]))]
            rho_new = [rho_k.item() for rho_k in rho_new]

        # normalize \rho
        rho = rho_old + rho_new
        rho = [r + 1e-10 for r in rho]  # TODO: to avoid zeros
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
            rho_old = [r + 1e-10 for r in rho_old]  # TODO: to avoid zeros
            rho_old = rho_old / np.sum(rho_old)
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho_old)]
            self.do_net_step(self.networks[k], x, log_losses=True)
        else:
            pass
            # logger.error('Index exceeds the length of components!')

        if self.new_net and self.new_net.n_points > (self.merge_burnin * 2):
            # Removal condition: net had 2 * self.merge_burnin points and never got chosen
            self.new_net = None

        if self.n_nets > 1:
            self.merge_nets()

        # TODO: reduce point set?
        wandb.log({
            "n_points": i + 1,
            "n_networks": self.n_nets,
            "biggest_net_size": max([net.n_points for net in self.networks]),
            "max_rho": np.max(self.rho_sum),
            "min_rho": np.min(self.rho_sum),
            "assigns": k,
        })

        self.reset_priors()
        return self.alpha

    def merge_nets(self):
        # Condition 1: check the components that has less data points than the self.merge_burnin condition,
        # and were not just picked
        most_recent_net = self.assigns[-1]
        for other_net in range(self.n_nets):
            if other_net == most_recent_net:
                continue
            if self.networks[other_net].n_points < self.merge_burnin * 2:
                self.merge_with_closest_net(other_net)

    def merge_with_closest_net(self, net):
        net_point_inds = [i for i in range(len(self.data)) if self.assigns[i] == net]
        net_logprob = th.cat([self.networks[net].log_likelihood(self.data[i][0], self.data[i][1], self.data[i][2]).reshape(-1)
                              for i in net_point_inds])
        klds = []
        for other_net in range(self.n_nets):
            if other_net == net:
                klds.append(np.inf)
                continue
            other_net_logprob = th.cat([self.networks[other_net].log_likelihood(self.data[i][0], self.data[i][1], self.data[i][2]).reshape(-1)
                                  for i in net_point_inds])
            kld = F.kl_div(net_logprob, other_net_logprob, log_target=True)
            klds.append(kld.item())
        closest_net = np.argmin(klds)
        if klds[closest_net] < self.merge_thresh:
            for ind in net_point_inds:
                self.do_net_step(self.networks[closest_net], self.data[ind])
                self.assigns[ind] = closest_net
            self.networks.pop(net)
            old_net_rho_sum = self.rho_sum[net]
            self.rho_sum[closest_net] += old_net_rho_sum
            self.rho_sum.pop(net)
            self.n_nets -= 1

    def forward(self, obs: th.Tensor, action: th.Tensor):
        most_likely_net = self.assigns[-1]
        with th.no_grad():
            return self.networks[most_likely_net](obs, action)

    def predict(self, s: np.array, a: np.array):
        if self.test_last_s is None:
            most_likely_net = self.assigns[-1]
        else:
            most_likely_net = np.argmax(self.test_priors, axis=0)
        with th.no_grad():
            predictions = self.networks[most_likely_net].model(th.cat((s, a), dim=1))
            return predictions

    def reset_priors(self):
        self.test_priors = self.rho_sum
        self.test_last_s = None
        self.test_last_a = None

    def pre_test_mpc(self, observation):
        if self.test_last_s is not None:
            new_priors = [self.test_priors[k] *
                      th.exp(self.networks[k].log_likelihood(self.test_last_s, self.test_last_a, observation))
                      for k in range(self.n_nets)]
            new_priors = [r.item() + 1e-10 for r in new_priors]  # TODO: to avoid zeros
            new_priors = new_priors / np.sum(new_priors)
            self.test_priors = [a + b for a, b in zip(self.test_priors, new_priors)]

    def post_test_mpc(self, observation, chosen_action):
        self.test_last_s = observation
        self.test_last_a = chosen_action

    def to(self, device=..., dtype=...,
           non_blocking: bool = ...):
        self.register_parameter("dummy", th.nn.Parameter(th.tensor([0], dtype=th.float32, device=device)))
        self.device = device
        for net in self.networks:
            net.to(device)
        return self
