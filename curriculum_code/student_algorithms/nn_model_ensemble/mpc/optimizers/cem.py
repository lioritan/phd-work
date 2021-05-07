
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch

from .optimizer import Optimizer


class CEMOptimizer(Optimizer):
    """A Pytorch-compatible CEM optimizer.
    """
    def __init__(self, sol_dim, popsize, upper_bound=None, lower_bound=None, max_iters=10, num_elites=100, epsilon=0.001, alpha=0.25, device=None):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        action_tiling_dim = self.sol_dim//upper_bound.shape[0]
        self.ub, self.lb = torch.tile(upper_bound,[action_tiling_dim]), torch.tile(lower_bound,[action_tiling_dim])
        self.epsilon, self.alpha = epsilon, alpha
        self.device = device

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.mean, self.var = None, None
        self.cost_function = None

    def setup(self, cost_function):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        self.cost_function = cost_function

        def sample_truncated_normal(shape, mu, sigma, a, b):
            '''
            Pytorch implementation of truncated normal distribution sampler

            Parameters:
            ----------
                @param numpy array or list - shape : size should be (popsize x sol_dim)
                @param numpy array or list - mu, sigma : size should be (sol_dim)
                @param tensor - a, b : lower bound and upper bound of sampling range, size should be (sol_dim)

            Return:
            ----------
                @param tensor - x : size should be (popsize x sol_dim)
            '''
            uniform = torch.rand(shape, device=self.device)
            normal = torch.distributions.normal.Normal(torch.tensor(0).to(device=self.device),
                                                       torch.tensor(1).to(device=self.device))

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            alpha_normal_cdf = normal.cdf(alpha)
            p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

            one = torch.tensor(1, dtype=p.dtype, device=self.device)
            epsilon = torch.tensor(1e-10, dtype=p.dtype, device=self.device)
            v = torch.clip(2 * p - 1, -one + epsilon, one - epsilon)
            x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
            x = torch.clamp(x, a[0], b[0])
            return x
        self.sample_trunc_norm = sample_truncated_normal

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """
        Optimizes the cost function using the provided initial candidate distribution parameters

        Parameters:
        ----------
            @param numpy array - init_mean, init_var: size should be (popsize x sol_dim)
            @param bool - use_pytorch: determine if use pytorch implementation
            @param bool - debug: if true, it will save some figures to help you find the best parameters

        Return:
        ----------
            @param numpy array - sol : size should be (sol_dim)
        """

        mean, var, t = init_mean, init_var, 0

        a, b = self.lb,self.ub
        size = [self.popsize, self.sol_dim]

        while (t < self.max_iters) and torch.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.minimum(torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2)), var)

            mu = mean
            sigma = torch.sqrt(constrained_var)
            samples = self.sample_trunc_norm(size, mu, sigma, a, b)

            costs = self.cost_function(samples)
            idx = torch.argsort(costs)
            elites = samples[idx][:self.num_elites]

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
            sol, solvar = mean, var
        return sol, solvar