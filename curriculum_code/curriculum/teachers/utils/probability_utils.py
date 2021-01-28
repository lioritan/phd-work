import numpy as np


def array_to_probability_dist(input_array):
    out_array = input_array.copy()

    has_negatives = (input_array < 0).any()
    if has_negatives:
        out_array += abs(input_array.min())

    total = out_array.sum()
    if total == 0:  # edge case - all negative and equal weight
        out_array += 1
        out_array = out_array / len(out_array)
    else:
        out_array = out_array / total
    return out_array
