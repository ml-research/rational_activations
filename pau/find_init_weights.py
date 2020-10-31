"""
find_init_weights.py
====================================
Finding the weights of the to map an specific activation function
"""

import json
import numpy as np
from .utils import fit_pau_to_base_function, plot_result, append_to_config_file
from .paus_py import PAU_version_A, PAU_version_B, PAU_version_C
import torch.nn.functional as F
import torch


def typed_input(text, type, choice_list = None):
    assert isinstance(text, str)
    while True:
        try:
            inp = input(text)
            typed_inp = type(inp)
            if choice_list is not None:
                assert typed_inp in choice_list
            break
        except ValueError:
            print(f"Please provide an type: {type}")
            continue
        except AssertionError:
            print(f"Please provide a value within {choice_list}")
            continue
    return typed_inp

FUNCTION = None

def find_weights(function):


    # To be changed by the function you want to approximate
    approx_name = input("approximated function name: ")
    FUNCTION = function
    def function_to_approx(x):
        # return np.heaviside(x, 0)
        x = torch.tensor(x)
        return FUNCTION(x)

    nd = typed_input("degree of the numerator P: ", int)
    dd = typed_input("degree of the denominator Q: ", int)
    degrees = (nd, dd)

    lb = typed_input("lower bound: ", float)
    ub = typed_input("upper bound: ", float)
    step = (ub - lb) / 100000
    x = np.arange(lb, ub, step)
    version = typed_input("PAU Version: ", str, ["A", "B", "C", "D"])
    if version == 'A':
        pau = PAU_version_A
    elif version == 'B':
        pau = PAU_version_B
    elif version == 'C':
        pau = PAU_version_C
    elif version == 'D':
        pau = PAU_version_B

    w_params, d_params = fit_pau_to_base_function(pau, function_to_approx, x,
                                                  degrees=degrees,
                                                  version=version)
    print(f"Found coeffient :\nP: {w_params}\nQ: {d_params}")
    plot = input("Do you want a plot of the result (y/n)") in ["y", "yes"]
    if plot:
        plot_result(x, pau(x, w_params, d_params), function_to_approx(x),
                    approx_name)
    params = {"version": version, "name": approx_name, "ub": ub, "lb": lb,
              "nd": nd, "dd": dd}
    append_to_config_file(params, approx_name, w_params, d_params)
