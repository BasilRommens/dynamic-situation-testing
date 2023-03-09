import numpy as np
import math


def is_nan(value):
    """
    :param value: value to check if nan
    :return: if value is nan

    >>> is_nan(float('nan'))
    True
    >>> is_nan(np.nan)
    True

    """
    if type(value) is str:
        return False
    return math.isnan(value)


def create_segment_name(segment_dict):
    segment_name = ''
    for idx, (key, (lo, hi)) in enumerate(segment_dict.items()):
        segment_name += f'{key}=[{lo},{hi}]'
        if idx != len(segment_dict) - 1:
            segment_name += ','
    return segment_name


def is_in_valid_tuples(valid_tuples, idx):
    for valid_tuple_idx, valid_tuple in enumerate(valid_tuples):
        if valid_tuple[0] == idx:
            return valid_tuple_idx
    return -1
