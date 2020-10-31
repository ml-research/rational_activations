import warnings
import json
import numpy as np
from numpy import zeros, inf
from scipy.optimize.optimize import OptimizeWarning
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy.optimize.minpack import leastsq, _wrap_jac
import matplotlib.pyplot as plt
import os
from pathlib import Path

np.random.seed(0)


def _wrap_func(func, xdata, ydata, degrees):
    def func_wrapped(params):
        params1 = params[:degrees[0]+1]
        params2 = params[degrees[0]+1:]
        return func(xdata, params1, params2) - ydata
    return func_wrapped


def _curve_fit(f, xdata, ydata, degrees, version, p0=None, absolute_sigma=False,
              method=None, jac=None, **kwargs):
    bounds = (-np.inf, np.inf)
    lb, ub = prepare_bounds(bounds, np.sum(degrees))
    if p0 is None:
        if version == "C":
            p0 = np.ones(np.sum(degrees)+2)
        else:
            p0 = np.ones(np.sum(degrees)+1)
    method = 'lm'

    ydata = np.asarray_chkfinite(ydata, float)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        xdata = np.asarray_chkfinite(xdata, float)

    # func = _wrap_func(xdata, ydata, degrees)  # Modification here  !!!
    func = _wrap_func(f, xdata, ydata, degrees)  # Modification here  !!!
    if callable(jac):
        jac = _wrap_jac(jac, xdata, None)
    elif jac is None and method != 'lm':
        jac = '2-point'

    if 'args' in kwargs:
        raise ValueError("'args' is not a supported keyword argument.")

    # Remove full_output from kwargs, otherwise we're passing it in twice.
    return_full = kwargs.pop('full_output', False)
    res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
    popt, pcov, infodict, errmsg, ier = res
    ysize = len(infodict['fvec'])
    cost = np.sum(infodict['fvec'] ** 2)
    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


def fit_pau_to_base_function(pau_func, ref_func, x, degrees=(5, 4), version="A"):
    y = ref_func(x)
    final_params = _curve_fit(pau_func, x, y, degrees=degrees, version=version,
                              maxfev=10000000)[0]
    return np.array(final_params[:degrees[0]+1]), np.array(final_params[degrees[0]+1:])


def plot_result(x_array, pau_array, target_array,
                original_func_name="Original function"):
    plt.plot(x_array, pau_array, label="PAU approx")
    plt.plot(x_array, target_array, label=original_func_name)
    plt.legend()
    plt.grid()
    plt.show()


def append_to_config_file(params, approx_name, w_params, d_params):
    ans = input("Do you want to store them in the json file ? (y/n)")
    if ans == "y" or ans == "yes":
        pau_full_name = f'PAU_version_{params["version"]}{params["nd"]}/{params["dd"]}'
        import ipdb; ipdb.set_trace()
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/paus_config.json') as json_file:
            paus_dict = json.load(json_file)  # pau_version -> approx_func
        approx_name = approx_name.lower()
        if pau_full_name in paus_dict:
            if approx_name in paus_dict[pau_full_name]:
                ans = input(f'PAU_{params["version"]} approximation of {approx_name} already exist.\
                              \nDo you want to replace it ? (y/n)')
                if not(ans == "y" or ans == "yes"):
                    print("Parameters not stored")
                    exit(0)
        else:
            paus_dict[pau_full_name] = {}
        paus_params = {"center": 0.0, "init_w_numerator": w_params.tolist(),
                       "init_w_denominator": d_params.tolist(),
                       "ub": params["ub"], "lb": params["lb"]}
        paus_dict[pau_full_name][approx_name] = paus_params
        with open(f'{cfd}/paus_config.json', 'w') as outfile:
            json.dump(paus_dict, outfile, indent=1)
        print("Parameters stored in paus_config.json")
    else:
        print("Parameters not stored")
        exit(0)


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
