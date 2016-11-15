import numpy as np


def merge(list1, list2):
    for a, b in zip(list1, list2):
        yield tuple(list(a) + [b])


def mean_of_non_zero(row):
    return np.mean(row[row!=0])