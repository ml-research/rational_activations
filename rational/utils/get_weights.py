import json
import os
from pathlib import Path
from .find_init_weights import find_weights
from .warnings import RationalImportError
import numpy as np
from termcolor import colored


known_functions = {
    "relu": lambda x: 0 if x < 0 else x,
    "leaky_relu": lambda x: x/100 if x < 0 else x,
    "lrelu": lambda x: x/100 if x < 0 else x,
    "normal": lambda x: 1/np.sqrt(2*np.pi) * np.exp(-.5*x**2),
}

def get_parameters(rational_version, degrees, approx_func):
    nd, dd = degrees
    if rational_version == 'T':
        return [1.] + [1, 1, 1], [1, 1, 1]
    if approx_func == "identity":
        return [0., 1.] + [0.] * (nd - 1), [0.] * dd
    elif approx_func == "ones":
        return [1.] * (nd + 1), [1.] * dd
    rational_full_name = f"Rational_version_{rational_version}{nd}/{dd}"
    config_file = '../rationals_config.json'
    config_file_dir = str(Path(os.path.abspath(__file__)).parent)
    with open(os.path.join(config_file_dir, config_file)) as json_file:
        rationals_dict = json.load(json_file)
    if rational_full_name not in rationals_dict:
        if approx_func.lower() in known_functions:
            msg = f"Found {approx_func} but haven't computed its rational approximation yet for degrees {degrees}.\nLet's do do it now:"
            print(colored(msg, "yellow"))
            find_weights(known_functions[approx_func.lower()], function_name=approx_func.lower(), degrees=degrees, version=rational_version)
            with open(os.path.join(config_file_dir, config_file)) as json_file:
                rationals_dict = json.load(json_file)
        else:
            config_not_found = f"{rational_full_name} approximating \"{approx_func}\" not found in {config_file}.\
            \nPlease add it (modify and run find_init_weights.py)"
            url = "https://rational-activations.readthedocs.io/en/latest/tutorials/tutorials.1_find_weights_for_initialization.html"
            raise RationalImportError(config_not_found, url)
    if approx_func not in rationals_dict[rational_full_name]:
        if approx_func.lower() in known_functions:
            msg = f"Found {approx_func} but haven't computed its rational approximation yet for degrees {degrees}.\nLet's do it now:"
            print(colored(msg, "yellow"))
            find_weights(known_functions[approx_func.lower()], function_name=approx_func.lower(), degrees=degrees, version=rational_version)
            with open(os.path.join(config_file_dir, config_file)) as json_file:
                rationals_dict = json.load(json_file)
        else:
            config_not_found = f"{rational_full_name} approximating {approx_func} not found in {config_file}.\
            \nPlease add it (modify and run find_init_weights.py)"
            url = "https://rational-activations.readthedocs.io/en/latest/tutorials/tutorials.1_find_weights_for_initialization.html"
            raise RationalImportError(config_not_found, url)
    params = rationals_dict[rational_full_name][approx_func]
    return params["init_w_numerator"], params["init_w_denominator"]
