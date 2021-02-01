from stable_baselines3.common.type_aliases import GymEnv
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box

from curriculum.teacher import Teacher
from typing import Tuple, Dict, Any

from curriculum.teachers.utils.knn_data import BufferedDataset
from curriculum.teachers.utils.task_space_utils import box_to_params, continuous_to_discrete, params_to_array


def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]


# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPComputer():
    def __init__(self, task_size, max_size=None, buffer_size=500):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    def compute_alp(self, task, reward):
        alp = 0
        lp = 0
        if len(self.alp_knn) > 5:
            # Compute absolute learning progress for new task

            # 1 - Retrieve closest previous task
            dist, idx = self.alp_knn.nn_y(task)

            # 2 - Retrieve corresponding reward
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])

            # 3 - Compute alp as absolute difference in reward
            lp = reward - closest_previous_task_reward
            alp = np.abs(lp)

        # Add to database
        self.alp_knn.add_xy(reward, task)
        return alp, lp


class AlpGmmTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)

        # Task space boundaries
        self.ordered_params = list(environment_parameters.parameters.keys())
        self.mins = np.array([environment_parameters.parameters[p_name].min_val for p_name in self.ordered_params])
        self.maxs = np.array([environment_parameters.parameters[p_name].max_val for p_name in self.ordered_params])

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = np.arange(2, 11, 1) if "potential_ks" not in teacher_parameters else teacher_parameters[
            "potential_ks"]
        # Restart new fit by initializing with last fit
        self.warm_start = False if "warm_start" not in teacher_parameters else teacher_parameters["warm_start"]
        # Fitness criterion when selecting best GMM among range of GMMs varying in number of Gaussians.
        self.gmm_fitness_fun = "aic" if "gmm_fitness_fun" not in teacher_parameters else teacher_parameters[
            "gmm_fitness_fun"]
        # Number of Expectation-Maximization trials when fitting
        self.nb_em_init = 1 if "nb_em_init" not in teacher_parameters else teacher_parameters['nb_em_init']
        # Number of episodes between two fit of the GMM
        self.fit_rate = 250 if "fit_rate" not in teacher_parameters else teacher_parameters['fit_rate']
        self.nb_random = self.fit_rate  # Number of bootstrapping episodes

        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = 0.2 if "random_task_ratio" not in teacher_parameters else teacher_parameters[
            "random_task_ratio"]
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)
        self.random_task_generator.seed(self.seed)

        # Maximal number of episodes to account for when computing ALP
        alp_max_size = None if "alp_max_size" not in teacher_parameters else teacher_parameters["alp_max_size"]
        alp_buffer_size = 500 if "alp_buffer_size" not in teacher_parameters else teacher_parameters["alp_buffer_size"]

        # Init ALP computer
        self.alp_computer = EmpiricalALPComputer(len(self.mins), max_size=alp_max_size, buffer_size=alp_buffer_size)

        self.tasks = []
        self.alps = []
        self.tasks_alps = []

        # Init GMMs
        self.potential_gmms = [self.init_gmm(k) for k in self.potential_ks]
        self.gmm = None

        # Boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_alps': [],
                   'tasks_lps': [], 'episodes': [], 'tasks_origin': []}

    def generate_task(self) -> Tuple[GymEnv, Dict[str, Any]]:
        task_origin = None
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Random task sampling
            new_task = self.random_task_generator.sample()
            task_origin = -1  # -1 = task originates from random sampling
        else:
            # ALP-based task sampling

            # 1 - Retrieve the mean ALP value of each Gaussian in the GMM
            self.alp_means = []
            for pos, _, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                self.alp_means.append(pos[-1])

            # 2 - Sample Gaussian proportionally to its mean ALP
            idx = proportional_choice(self.alp_means, eps=0.0)
            task_origin = idx

            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
            new_task = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        # boring book-keeping
        self.bk['tasks_origin'].append(task_origin)
        task_params = box_to_params(self.ordered_params, new_task)
        task_params = continuous_to_discrete(self.env_wrapper, self.ordered_params, task_params)
        return self.env_wrapper.create_env(task_params), task_params

    def update_teacher_policy(self):
        task, reward = self.history[-1]
        task_array = params_to_array(self.ordered_params, task)
        self.tasks.append(task_array)

        # Compute corresponding ALP
        alp, lp = self.alp_computer.compute_alp(task_array, reward)
        self.alps.append(alp)

        # Concatenate task vector with ALP dimension
        self.tasks_alps.append(np.array(task_array.tolist() + [self.alps[-1]]))

        if len(self.tasks) >= self.nb_random:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # 1 - Retrieve last <fit_rate> (task, reward) pairs
                cur_tasks_alps = np.array(self.tasks_alps[-self.fit_rate:])

                # 2 - Fit batch of GMMs with varying number of Gaussians
                self.potential_gmms = [g.fit(cur_tasks_alps) for g in self.potential_gmms]

                # 3 - Compute fitness and keep best GMM
                fitnesses = []
                if self.gmm_fitness_fun == 'bic':  # Bayesian Information Criterion
                    fitnesses = [m.bic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aic':  # Akaike Information Criterion
                    fitnesses = [m.aic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aicc':  # Modified AIC
                    n = self.fit_rate
                    fitnesses = []
                    for l, m in enumerate(self.potential_gmms):
                        k = self.get_nb_gmm_params(m)
                        penalty = (2 * k * (k + 1)) / (n - k - 1)
                        fitnesses.append(m.aic(cur_tasks_alps) + penalty)
                else:
                    raise NotImplementedError
                self.gmm = self.potential_gmms[np.argmin(fitnesses)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['tasks_alps'] = self.tasks_alps
                self.bk['tasks_lps'].append(lp)
                self.bk['episodes'].append(len(self.tasks))

    def init_gmm(self, nb_gaussians):
        return GMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                   warm_start=self.warm_start, n_init=self.nb_em_init)

    def get_nb_gmm_params(self, gmm):
        # assumes full covariance
        # see https://stats.stackexchange.com/questions/229293/the-number-of-parameters-in-gaussian-mixture-model
        nb_gmms = gmm.get_params()['n_components']
        d = len(self.mins)
        params_per_gmm = (d * d - d) / 2 + 2 * d + 1
        return nb_gmms * params_per_gmm - 1
