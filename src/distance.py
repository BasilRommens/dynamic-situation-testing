import time

import numpy as np
from functools import lru_cache
from helper import is_nan


def total_distance(t_1, t_2, all_tuples, attribute_types, ranked_values_per_ord,
                   m_dict, s_dict, decision_attr_idx, protected_attr_idcs):
    """
    determine the total distance between two tuples
    :param t_1: first tuple
    :param t_2: second tuple
    :param all_tuples: all the tuples in the dataset
    :param attribute_types: the types at each index of a tuple
    :param ranked_values_per_ord: the ranked values for each ordinal attribute
    :return: total distance between the two tuples
    """
    attributes = list(all_tuples.columns)

    # the sum placed in the numerator for each tuple-value
    summation = 0

    start = time.time()

    # go over all tuple values
    for idx, (v_1, v_2) in enumerate(zip(t_1, t_2)):
        # if the attribute is the decision attribute continue
        if idx == decision_attr_idx:
            continue
        # if the attribute is a protected attribute continue
        if idx in protected_attr_idcs:
            continue

        attribute_type = attribute_types[
            attributes[idx]]  # determine attribute type

        # determine the distance metric based on the attribute type
        if attribute_type == 'interval':
            summation += interval_distance(v_1, v_2, m_dict[idx], s_dict[idx])
        elif attribute_type == 'nominal':
            summation += nominal_distance(v_1, v_2)
        elif attribute_type == 'ordinal':
            ranked_values = ranked_values_per_ord[attributes[idx]]

            summation += ordinal_distance(v_1, v_2, len(ranked_values))

    # compute the average distance over all tuple values
    distance = summation / len(t_1)
    end = time.time()
    print(f"Total time spent in distance: f{end - start}s")

    return distance


def interval_distance(v_1: float, v_2: float, m: float, s: float):
    """
    determine the distance between two interval values
    :param v_1: first interval value
    :param v_2: second interval value
    :param m: mean of all interval values
    :param s: standard deviation of all interval values
    :return: interval distance
    """
    # in case that either v_1 or v_2 is None return 3 by default
    if is_nan(v_1) or is_nan(v_2):
        return 3

    # determine the distance
    d = abs(v_1 - v_2) / s

    return d


@lru_cache(maxsize=None)
def nominal_distance(v_1, v_2):
    """
    determine the distance between two nominal values
    :param v_1: first nominal value
    :param v_2: second nominal value
    :return: nominal distance (0 if equal, 1 otherwise)
    """
    # in case that either v_1 or v_2 is None
    # return 1 by default
    if is_nan(v_1) or is_nan(v_2):
        return 1

    # determine the distance
    if v_1 == v_2:  # if the values are the same return 0
        return 0
    else:  # if the values are different return 1
        return 1


def interval_scale(idx, M):
    """
    formula for interval scaling of ordinal values
    :param idx: the idx of the ordinal value
    :param M: the length of the ranked values
    :return: the interval scaled value
    """
    return (idx - 1) / (M - 1)


@lru_cache(maxsize=None)
def ordinal_distance(v_idx_1, v_idx_2, ranked_values_len):
    """
    determine the distance between two ordinal values
    :param v_idx_1: index of first ordinal value
    :param v_idx_2: index of second ordinal value
    :param ranked_values: the ranked ordinal values in list form
    :return: distance between the ordinal values
    """
    is_nan_v_idx_1 = is_nan(v_idx_1)
    is_nan_v_idx_2 = is_nan(v_idx_2)
    # if both indexes are none then return 1
    if is_nan_v_idx_1 and is_nan_v_idx_2:
        return 1
    # otherwise in case that only one of them is none
    # use a formula to determine the distance
    elif is_nan_v_idx_1 and not is_nan_v_idx_2:
        m_2 = interval_scale(v_idx_2, ranked_values_len)
        d = max(m_2, 1 - m_2)
        return d
    elif is_nan_v_idx_2 and not is_nan_v_idx_1:
        m_1 = interval_scale(v_idx_1, ranked_values_len)
        d = max(m_1, 1 - m_1)
        return d
    # in the last case use the difference in interval scale
    else:
        m_1 = interval_scale(v_idx_1, ranked_values_len)
        m_2 = interval_scale(v_idx_2, ranked_values_len)
        d = abs(m_1 - m_2)
        return d
