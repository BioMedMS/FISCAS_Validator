import numpy as np

#TODO replace makeshift solution
def sum_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))

def mean_absolute_percentage_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-12)