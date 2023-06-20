import itertools
import time
from multiprocessing import Pool, cpu_count
import scipy.spatial as scs

import numpy as np
from functools import lru_cache

import distance_fast
from helper import is_nan


def total_distance(t_1, t_2, attributes, attribute_types, ranked_values_per_ord,
                   m_dict, s_dict, decision_attr_idx, protected_attr_idcs):
    """
    determine the total distance between two tuples
    :param t_1: first tuple
    :param t_2: second tuple
    :param attributes: the attributes of the tuples
    :param attribute_types: the types at each index of a tuple
    :param ranked_values_per_ord: the ranked values for each ordinal attribute
    :return: total distance between the two tuples
    """
    # the sum placed in the numerator for each tuple-value
    summation = 0

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
    if v_1 is None or v_2 is None:
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
    if v_1 is None or v_2 is None:
        return 1

    # determine the distance
    if v_1 == v_2:  # if the values are the same return 0
        return 0
    else:  # if the values are different return 1
        return 1


@lru_cache(maxsize=None)
def ordinal_distance(v_1, v_2, ranked_values_len):
    """
    determine the distance between two ordinal values
    :param v_1: first ordinal value
    :param v_2: second ordinal value
    :param ranked_values: the ranked ordinal values in list form
    :return: distance between the ordinal values
    """
    is_nan_v_1 = v_1 is None
    is_nan_v_2 = v_2 is None
    # if both indexes are none then return 1
    if is_nan_v_1 and is_nan_v_2:
        return 1
    # in case that only one of them is none
    # use a formula to determine the distance
    elif is_nan_v_1 and not is_nan_v_2:
        m_2 = interval_scale(v_2, ranked_values_len)
        d = max(m_2, 1 - m_2)
        return d
    elif is_nan_v_2 and not is_nan_v_1:
        m_1 = interval_scale(v_1, ranked_values_len)
        d = max(m_1, 1 - m_1)
        return d
    # last case: use the difference in interval scale
    else:
        m_1 = interval_scale(v_1, ranked_values_len)
        m_2 = interval_scale(v_2, ranked_values_len)
        d = abs(m_1 - m_2)
        return d


@lru_cache(maxsize=80)
def interval_scale(val, M):
    """
    formula for interval scaling of ordinal values
    :param val: the value of the ordinal value
    :param M: the length of the ranked values
    :return: the interval scaled value
    """
    return (val - 1) / (M - 1)


def pdist(tuples, tuple_idx_from, metric):
    """
    calculates the distance between a tuple and a set of tuples
    :param tuples: all the tuples to calculate the distance for
    :param tuple_idx_from: the tuple index from which to calculate the distance
    :param metric: the distance function
    :return: the distances between the tuple and the set of tuples
    """
    # get the tuple from which to calculate the distance
    tuple_from = tuples[tuple_idx_from]

    # calculate the distances
    distances = np.zeros(len(tuples) - tuple_idx_from - 1)
    # start = time.time()
    for idx, tuple_idx_to in enumerate(range(tuple_idx_from + 1, len(tuples))):
        tuple_to = tuples[tuple_idx_to]
        distances[idx] = metric(tuple_from, tuple_to)
    # print(time.time() - start)

    return distances


def args_gen(tuple_len):
    """
    generates the arguments for the pdist function
    :return: the arguments
    """
    for i in range(tuple_len):
        yield i, [j for j in range(i, tuple_len)]


def ppdist(tuples, metric):
    """
    Create a pairwise distance matrix in parallel
    :param tuples: the tuples to create the pairwise distance matrix for
    :param metric: the distance function to use
    :return: distance matrix
    """
    # make a list of arguments, where the first argument is the first tuple and
    # the second argument is a set of tuples
    args = [i for i in range(len(tuples))]

    # create a pool of processes
    with Pool(processes=cpu_count()) as pool:
        # map the distance function to the arguments
        pdist_fun = PDist(tuples, metric)
        flat_dist_matrix = pool.map(pdist_fun, args)

    return list(itertools.chain(*flat_dist_matrix))


def transform_tuples(tuples, attributes, attribute_types, ranked_values_per_ord,
                     m_dict, s_dict, decision_attr_idx, protected_attr_idcs):
    """
    transform the tuples so that less calculations are done each time
    :param tuples: the tuples to transform
    :param attributes: the names of the attributes
    :param attribute_types: the types of the attributes in dict form
    :param ranked_values_per_ord: the ranked values for each ordinal attribute
    :param m_dict: the dict of the mean values
    :param s_dict: the dict of the standard deviation values
    :param decision_attr_idx: the decision attribute index
    :param protected_attr_idcs: the indices of the protected attributes
    :return: the transformed tuples
    """
    for idx, col in enumerate(tuples.columns):
        # if the attribute is the decision attribute continue
        if idx == decision_attr_idx:
            tuples = tuples.drop(col, axis=1)
            continue
        # if the attribute is a protected attribute continue
        if idx in protected_attr_idcs:
            tuples = tuples.drop(col, axis=1)
            continue

        attribute_type = attribute_types[col]  # determine attribute type

        # determine the distance metric based on the attribute type
        if attribute_type == 'interval':
            tuples[col] = tuples[col].apply(lambda x: x / s_dict[idx])
        elif attribute_type == 'ordinal':
            tuples[col] = tuples[col].apply(
                lambda x: interval_scale(x, len(ranked_values_per_ord[col])))
    return tuples


class PDist:
    def __init__(self, tuples, metric):
        self.tuples = tuples
        self.metric = metric

    def __call__(self, tuple_idx_from):
        return pdist(self.tuples, tuple_idx_from, self.metric)


class Metric:
    def __init__(self, attributes, attribute_types, ranked_values, m_dict,
                 s_dict, decision_attr_idx, protected_attr_idcs):
        self.attributes = attributes
        self.attribute_types = attribute_types
        self.ranked_values = ranked_values
        self.m_dict = m_dict
        self.s_dict = s_dict
        self.decision_attr_idx = decision_attr_idx
        self.protected_attr_idcs = protected_attr_idcs

    def __call__(self, u, v):
        return distance_fast.ugly_total_distance(u, v, self.s_dict,
                                                 self.attribute_types)
