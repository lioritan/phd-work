from environment.environment_parameter import ContinuousParameter


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
