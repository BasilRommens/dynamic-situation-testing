import numpy as np


def total_distance(t_1, t_2, all_tuples, attributes, attribute_types,
                   ranked_values_per_ord, decision_attribute,
                   protected_attributes):
    """
    determine the total distance between two tuples
    :param t_1: first tuple
    :param t_2: second tuple
    :param all_tuples: all the tuples in the dataset
    :param attribute_types: the types at each index of a tuple
    :param ranked_values_per_ord: the ranked values for each ordinal attribute
    :param decision_attribute: decision attribute to be ignored for distance
    :param protected_attributes: protected attributes to be ignored for distance
    :return: total distance between the two tuples
    """
    # the sum placed in the numerator for each tuple-value
    summation = 0
    # go over all tuple values
    for idx, (v_1, v_2) in enumerate(zip(t_1, t_2)):
        # if the attribute is the decision attribute continue
        if idx == attributes.index(list(decision_attribute.keys())[0]):
            continue
        # if the attribute is a protected attribute continue
        if idx in [attributes.index(protected_attribute) for protected_attribute
                   in protected_attributes.keys()]:
            continue

        attribute_type = attribute_types[
            attributes[idx]]  # determine attribute type

        # determine the distance metric based on the attribute type
        if attribute_type == 'interval':
            all_interval = list(map(lambda x: x[idx], all_tuples))
            m = np.average(all_interval)
            s = np.std(all_interval)

            summation += interval_distance(v_1, v_2, m, s)
        elif attribute_type == 'nominal':
            summation += nominal_distance(v_1, v_2)
        elif attribute_type == 'ordinal':
            ranked_values = ranked_values_per_ord[attributes[idx]]
            v_idx_1, v_idx_2 = v_1, v_2

            summation += ordinal_distance(v_idx_1, v_idx_2, ranked_values)

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
    # in case that either v_1 or v_2 is None
    # return 3 by default
    if v_1 is None or v_2 is None:
        return 3

    # determine the z-score for each value
    z_1 = (v_1 - m) / s
    z_2 = (v_2 - m) / s

    # determine the distance
    d = abs(z_1 - z_2)

    return d


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


def interval_scale(idx, ranked_values: list[float]):
    """
    formula for interval scaling of ordinal values
    :param idx: the idx of the ordinal value
    :param ranked_values: the ranked values in list form
    :return: the interval scaled value
    """
    M = len(ranked_values)

    return (idx - 1) / (M - 1)


def ordinal_distance(v_idx_1, v_idx_2, ranked_values):
    """
    determine the distance between two ordinal values
    :param v_idx_1: index of first ordinal value
    :param v_idx_2: index of second ordinal value
    :param ranked_values: the ranked ordinal values in list form
    :return: distance between the ordinal values
    """
    # if both indexes are none then return 1
    if v_idx_1 is None and v_idx_2 is None:
        return 1
    # otherwise in case that only one of them is none
    # use a formula to determine the distance
    elif v_idx_1 is None and v_idx_2 is not None:
        m_2 = interval_scale(v_idx_2, ranked_values)
        d = max(m_2, 1 - m_2)
        return d
    elif v_idx_2 is None and v_idx_1 is not None:
        m_1 = interval_scale(v_idx_1, ranked_values)
        d = max(m_1, 1 - m_1)
        return d
    # in the last case use the difference in interval scale
    else:
        m_1 = interval_scale(v_idx_1, ranked_values)
        m_2 = interval_scale(v_idx_2, ranked_values)
        d = abs(m_1 - m_2)
        return d
