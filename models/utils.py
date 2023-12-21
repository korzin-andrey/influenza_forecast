import numpy as np


def real_to_abs(array, rho):
    array_out = []
    for t in range(0, len(array)):
        array_out.append(array[t] * 10000.0 / float(rho))
    return np.array(array_out)
