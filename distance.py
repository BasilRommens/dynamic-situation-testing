def total_distance(t_1, t_2):
    pass


def interval_distance(v_1, v_2, m, s):
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
    M = len(ranked_values)
    return (idx - 1) / (M - 1)


def ordinal_distance(v_idx_1, v_idx_2, ranked_values):
    if v_idx_1 is None and v_idx_2 is None:
        return 1
    elif v_idx_1 is None:
        m_2 = interval_scale(v_idx_2, ranked_values)
        d = max(m_2, 1 - m_2)
        return d
    elif v_idx_2 is None:
        m_1 = interval_scale(v_idx_1, ranked_values)
        d = max(m_1, 1 - m_1)
        return d
    else:
        m_1 = interval_scale(v_idx_1, ranked_values)
        m_2 = interval_scale(v_idx_2, ranked_values)
        d = abs(m_1 - m_2)
        return d
