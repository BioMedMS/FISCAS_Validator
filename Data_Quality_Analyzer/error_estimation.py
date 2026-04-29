import numpy as np

def symmetric_l1_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + np.sum(np.abs(b)))