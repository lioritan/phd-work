from environment.environment_parameter import ContinuousParameter
import scipy.signal


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
