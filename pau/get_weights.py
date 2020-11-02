import json
import os
from pathlib import Path


def get_parameters(pau_version, degrees, approx_func):
    nd, dd = degrees
    pau_full_name = f"PAU_version_{pau_version}{nd}/{dd}"
    config_file = 'paus_config.json'
    config_file_dir = str(Path(os.path.abspath(__file__)).parent)
    with open(os.path.join(config_file_dir, config_file)) as json_file:
        paus_dict = json.load(json_file)
    config_not_found = f"{pau_full_name} approximating {approx_func} not found in {config_file}.\
                          \nPlease add it (modify and run find_init_weights.py)"
    if pau_full_name not in paus_dict:
        print(config_not_found)
        exit(1)
    if approx_func not in paus_dict[pau_full_name]:
        print(config_not_found)
        exit(1)
    params = paus_dict[pau_full_name][approx_func]
    return params["center"], params["init_w_numerator"], params["init_w_denominator"]
