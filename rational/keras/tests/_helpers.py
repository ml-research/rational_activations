import numpy as np
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

"""
This file contains methods that are useful for multiple test files in this directory
"""


def activation(func, data):
    """
    apply activation function to data

    :param func: activation function
    :param data: data to be applied
    """
    if func == leaky_relu:
        return np.array(func(data, alpha=0.01))
    else:
        return np.array(func(data))
