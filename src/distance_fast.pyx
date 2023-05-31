import math

cpdef ugly_total_distance(t_1, t_2, attribute_types):
    """
    determine the total distance between two tuples
    :param t_1: first tuple
    :param t_2: second tuple
    :param attribute_types: the types at each index of a tuple
    :return: total distance between the two tuples
    """
    # the sum placed in the numerator for each tuple-value
    cdef float summation = 0.0
    v_1 = None
    v_2 = None
    cdef int attribute_type = 0

    # go over all tuple values
    for idx in range(len(t_1)):
        v_1 = t_1[idx]
        v_2 = t_2[idx]

        attribute_type = attribute_types[idx]  # determine attribute type

        # determine the distance metric based on the attribute type
        if attribute_type == 0:  # interval
            summation += 3.0 if v_1 is None or v_2 is None else abs(v_1 - v_2)
        elif attribute_type == 1:  # nominal
            summation += 1.0 if v_1 is None else float(v_1 != v_2)
        elif attribute_type == 2:  # ordinal
            is_nan_v_1 = v_1 is None
            is_nan_v_2 = v_2 is None
            # if both indexes are none then return 1
            if is_nan_v_1 and is_nan_v_2:
                summation += 1.0
            # in case that only one of them is none
            # use a formula to determine the distance
            elif is_nan_v_1:
                summation += max(v_2, 1.0 - v_2)
            elif is_nan_v_2:
                summation += max(v_1, 1.0 - v_1)
            # last case: use the difference in interval scale
            else:
                summation += abs(v_1 - v_2)

    # compute the average distance over all tuple values
    return summation / len(t_1)
