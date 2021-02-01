from environment.environment_parameter import ContinuousParameter, DiscreteParameter
import scipy.signal
import numpy as np


def has_similar_task(task, params, task_bins, continuous_sensativity):
    continuous_params = [k for k, v in params.items() if isinstance(v, ContinuousParameter)]
    for candidate in task_bins:
        is_similar = True
        for param_name in params.keys():
            if param_name in continuous_params:
                value_bin = continuous_sensativity * (params[param_name].max_val - params[param_name].min_val)
                if abs(candidate[param_name] - task[param_name]) > value_bin:  # too far apart
                    is_similar = False
                    break
            else:
                if candidate[param_name] != task[param_name]:  # not the same
                    is_similar = False
                    break
        if is_similar:
            return True
    return False


def box_to_params(ordered_params, box_sample):
    return {ordered_params[i]: box_sample[i] for i in range(len(ordered_params))}


def params_to_array(ordered_params, params):
    return np.array([params[p] for p in ordered_params])


def continuous_to_discrete(env_params, ordered_params, task):
    new_task = task.copy()
    for i, param_name in enumerate(ordered_params):
        if isinstance(env_params.parameters[param_name], DiscreteParameter):
            new_task[i] = round(new_task[i])
    return new_task


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
