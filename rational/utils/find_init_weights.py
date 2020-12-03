"""
find_init_weights.py
====================================
Finding the weights of the to map an specific activation function
"""

import json
import numpy as np
from .utils import fit_rational_to_base_function
import matplotlib.pyplot as plt
import torch
import os
from rational.numpy.rationals import Rational_version_A, Rational_version_B, \
    Rational_version_C


def plot_result(x_array, rational_array, target_array,
                original_func_name="Original function"):
    plt.plot(x_array, rational_array, label="Rational approx")
    plt.plot(x_array, target_array, label=original_func_name)
    plt.legend()
    plt.grid()
    plt.show()


def append_to_config_file(params, approx_name, w_params, d_params):
    ans = input("Do you want to store them in the json file ? (y/n)")
    if ans == "y" or ans == "yes":
        rational_full_name = f'Rational_version_{params["version"]}{params["nd"]}/{params["dd"]}'
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/rationals_config.json') as json_file:
            rationals_dict = json.load(json_file)  # rational_version -> approx_func
        approx_name = approx_name.lower()
        if rational_full_name in rationals_dict:
            if approx_name in rationals_dict[rational_full_name]:
                ans = input(f'Rational_{params["version"]} approximation of {approx_name} already exist.\
                              \nDo you want to replace it ? (y/n)')
                if not (ans == "y" or ans == "yes"):
                    print("Parameters not stored")
                    exit(0)
        else:
            rationals_dict[rational_full_name] = {}
        rationals_params = {"init_w_numerator": w_params.tolist(),
                            "init_w_denominator": d_params.tolist(),
                            "ub": params["ub"], "lb": params["lb"]}
        rationals_dict[rational_full_name][approx_name] = rationals_params
        with open(f'{cfd}/rationals_config.json', 'w') as outfile:
            json.dump(rationals_dict, outfile, indent=1)
        print("Parameters stored in rationals_config.json")
    else:
        print("Parameters not stored")
        exit(0)


def typed_input(text, type, choice_list=None):
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


def find_weights(function, version=None, degrees=None):
    # To be changed by the function you want to approximate
    approx_name = input("approximated function name: ")
    FUNCTION = function

    def function_to_approx(x):
        # return np.heaviside(x, 0)
        x = torch.tensor(x)
        return FUNCTION(x)

    if degrees is None:
        nd = typed_input("degree of the numerator P: ", int)
        dd = typed_input("degree of the denominator Q: ", int)
        degrees = (nd, dd)

    print("On what range should the function be approximated ?")
    lb = typed_input("lower bound: ", float)
    ub = typed_input("upper bound: ", float)
    step = (ub - lb) / 100000
    x = np.arange(lb, ub, step)
    if version is None:
        version = typed_input("Rational Version: ", str, ["A", "B", "C", "D"])
    if version == 'A':
        rational = Rational_version_A
    elif version == 'B':
        rational = Rational_version_B
    elif version == 'C':
        rational = Rational_version_C
    elif version == 'D':
        rational = Rational_version_B

    w_params, d_params = fit_rational_to_base_function(rational, function_to_approx, x,
                                                       degrees=degrees,
                                                       version=version)
    print(f"Found coeffient :\nP: {w_params}\nQ: {d_params}")
    plot = input("Do you want a plot of the result (y/n)") in ["y", "yes"]
    if plot:
        plot_result(x, rational(x, w_params, d_params), function_to_approx(x),
                    approx_name)
    params = {"version": version, "name": approx_name, "ub": ub, "lb": lb,
              "nd": nd, "dd": dd}
    append_to_config_file(params, approx_name, w_params, d_params)
    return w_params, d_params
