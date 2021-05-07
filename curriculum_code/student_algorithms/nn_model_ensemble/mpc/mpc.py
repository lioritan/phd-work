import numpy as np
from tqdm import trange, tqdm
from .optimizers import RandomShootingOptimizer, CEMOptimizer
import copy
import torch as th


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomShootingOptimizer}

    def __init__(self, mpc_config):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.device = conf["device"]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = th.tensor(conf["action_low"], device=self.device)  # array (dim,)
        self.action_high = th.tensor(conf["action_high"], device=self.device)  # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]
        self.particle = conf["particle"]

        self.cost_func = None

        self.init_mean = th.tensor([conf["init_mean"]] * self.horizon, device=self.device)
        self.init_var = th.tensor([conf["init_var"]] * self.horizon, device=self.device)

        if len(self.action_low) == 1:  # auto fill in other dims
            self.action_low = th.tile(self.action_low, [self.action_dim])
            self.action_high = th.tile(self.action_high, [self.action_dim])

        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon * self.action_dim,
                                                   popsize=self.popsize,
                                                   upper_bound=th.tensor(conf["action_high"], device=self.device),
                                                   lower_bound=th.tensor(conf["action_low"], device=self.device),
                                                   max_iters=conf["max_iters"],
                                                   num_elites=conf["num_elites"],
                                                   epsilon=conf["epsilon"],
                                                   alpha=conf["alpha"],
                                                   device=self.device,)

        self.set_cost_func(conf.get("cost_function"))
        self.reset()

    def set_cost_func(self, cost_func):
        self.cost_func = cost_func
        self.optimizer.setup(self.action_cost_function)

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        # print('set init mean to 0')
        self.prev_sol = th.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = th.tile(th.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, task, model, state):
        """
        :param state: task, model, (numpy array) current state
        :return: (float) optimal action
        """
        self.task = task
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = th.cat([soln[self.action_dim:], th.zeros(self.action_dim, device=self.device)])

        action = soln[:self.action_dim]
        return action

    def preprocess(self, state):
        pass

    def action_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        actions = actions.reshape((-1, self.horizon, self.action_dim))  # [pop size, horizon, action_dim]
        actions = th.tile(actions, (self.particle, 1, 1))

        costs = th.zeros(self.popsize * self.particle, device=self.device)
        state = th.Tensor.repeat(self.state.reshape(1, -1), (self.popsize * self.particle, 1))

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            # the output of the prediction model is [state_next - state]
            state_next = self.model.predict(state, action)

            cost = self.cost_func(state_next, action)  # compute cost
            costs += cost * self.gamma ** t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = th.mean(costs.reshape((self.particle, -1)), dim=0)
        return costs
