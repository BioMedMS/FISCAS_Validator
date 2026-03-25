import numpy as np

#TODO replace makeshift solution
def sum_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))