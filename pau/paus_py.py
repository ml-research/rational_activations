import numpy as np
import time


def PAU_version_A(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.ones_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += np.abs(d_array[i] * xi)
    return P/Q


def PAU_version_B(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += d_array[i] * xi
    Q = np.abs(Q) + np.ones_like(Q)
    return P/Q


def PAU_version_C(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        Q += d_array[i] * xi  # Here b0 is considered
        xi *= x
    Q = np.abs(Q) + np.full_like(Q, 0.1)
    return P/Q
