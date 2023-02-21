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


if __name__ == '__main__':
    pass
