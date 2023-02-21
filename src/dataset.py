import numpy as np


def create_gaussian(dim, n):
    anchor_point = np.random.randint(low=-100, high=100, size=dim)
    new_points = list()
    for _ in range(n):
        new_point = np.random.normal(size=dim)
        new_point += anchor_point
        new_points.append(new_point)
    return new_points, anchor_point


if __name__ == '__main__':
    pts = create_gaussian(6, 100)
