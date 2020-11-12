import json
import os
from pathlib import Path


def get_parameters(rational_version, degrees, approx_func):
    nd, dd = degrees
    rational_full_name = f"Rational_version_{rational_version}{nd}/{dd}"
    config_file = 'rationals_config.json'
    config_file_dir = str(Path(os.path.abspath(__file__)).parent)
    with open(os.path.join(config_file_dir, config_file)) as json_file:
        rationals_dict = json.load(json_file)
    config_not_found = f"{rational_full_name} approximating {approx_func} not found in {config_file}.\
                          \nPlease add it (modify and run find_init_weights.py)"
    if rational_full_name not in rationals_dict:
        print(config_not_found)
        exit(1)
    if approx_func not in rationals_dict[rational_full_name]:
        print(config_not_found)
        exit(1)
    params = rationals_dict[rational_full_name][approx_func]
    return params["init_w_numerator"], params["init_w_denominator"]
