import numpy as np
from tqdm import trange, tqdm
from .optimizers import RandomShootingOptimizer, CEMOptimizer
import copy


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomShootingOptimizer}

    def __init__(self, mpc_config):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"])  # array (dim,)
        self.action_high = np.array(conf["action_high"])  # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]
        self.particle = conf["particle"]

        self.cost_func = None

        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1:  # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim])

        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon * self.action_dim,
                                                   popsize=self.popsize,
                                                   upper_bound=np.array(conf["action_high"]),
                                                   lower_bound=np.array(conf["action_low"]),
                                                   max_iters=conf["max_iters"],
                                                   num_elites=conf["num_elites"],
                                                   epsilon=conf["epsilon"],
                                                   alpha=conf["alpha"])

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
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, task, model, state, ground_truth=False):
        '''
        :param state: task, model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.task = task
        self.model = model
        self.state = state
        self.ground_truth = ground_truth

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        else:
            pass

        # since cartpole has only one action dimension
        action = soln[0]
        return action

    def preprocess(self, state):
        # given state return observation
        # state = (x, x_dot, theta, theta_dot)
        # obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])
        obs = np.concatenate([state[:, 0:1], state[:, 1:2],
                              np.cos(state[:, 2:3]), np.sin(state[:, 2:3]), state[:, 3:]], axis=1)
        # print('obs shape ', obs.shape)
        return obs

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
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize * self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize * self.particle, axis=0)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            # state_next = self.model.predict(self.index, state, action)+state  # numpy array (batch_size x state dim)
            # the output of the prediction model is [state_next - state]
            if not self.ground_truth:
                state_next = self.model.predict(state, action) + state
                # state_next = self.model.predict( self.preprocess(state), action) + state  # numpy array (batch_si
            else:
                # change to ground truth one
                state_next = []
                for i in range(state.shape[0]):
                    self.task.set_state(state[i])
                    state_next_i, reward, done, info = self.task.step(action[i])
                    state_next.append(state_next_i)
                state_next = np.array(state_next)

            cost = self.cost_func(state_next, action)  # compute cost
            costs += cost * self.gamma ** t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs
