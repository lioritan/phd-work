import math

import torch
from torch.optim import Optimizer


# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md
class SimpleSGLDPriorSampling(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics for low temperature and prior sampling for high temperatures
    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 beta=1.0,
                 num_burn_in_steps=0) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        beta : float, optional
            Exponential parameter of gibbs (should ideally be sqrt(num samples))
            Default: `1.0`
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `0`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, beta=beta,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)
        self.last_noise = []

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        self.last_noise = []

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                beta = group["beta"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0

                #  }}} State initialization #

                state["iteration"] += 1

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = torch.ones_like(parameter)
                else:
                    sigma = torch.zeros_like(parameter)

                # if beta is high, we want posterior sampling, if it's lo we want prior sampling
                noise_variance = math.sqrt(lr / beta) if beta >= 1.0 else math.sqrt(beta)

                scaled_noise = (
                        torch.normal(
                            mean=torch.zeros_like(gradient),
                            std=torch.ones_like(gradient) * noise_variance
                        ) * sigma
                )  # Noise is N(0, sqrt(lr/beta))

                # Note: noise is scaled to gradient to make it less random
                scale_noise_to_grad = False
                if scale_noise_to_grad:
                    scaled_noise = scaled_noise / torch.norm(gradient)

                if beta >= 1.0:
                    parameter.data.add_(-lr * gradient + scaled_noise)
                else:
                    parameter.data.add_(scaled_noise)
                self.last_noise.append(scaled_noise)

        return loss
