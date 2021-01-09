"""
This file contains methods that are useful for multiple test files in this directory
"""
import numpy as np
from tensorflow.nn import leaky_relu


def activation(func, data):
    """
    apply activation function to data

    :param func: activation function
    :param data: data to be applied
    """
    if func == leaky_relu:
        return np.array(func(data, alpha=0.01))
    return np.array(func(data))
