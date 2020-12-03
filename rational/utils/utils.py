import warnings
import numpy as np
from numpy import zeros, inf

# np.random.seed(0)


def _wrap_func(func, xdata, ydata, degrees):
    def func_wrapped(params):
        params1 = params[:degrees[0]+1]
        params2 = params[degrees[0]+1:]
        return func(xdata, params1, params2) - ydata
    return func_wrapped


def _curve_fit(f, xdata, ydata, degrees, version, p0=None, absolute_sigma=False,
               method=None, jac=None, **kwargs):
    from scipy.optimize.optimize import OptimizeWarning
    from scipy.optimize._lsq.least_squares import prepare_bounds
    from scipy.optimize.minpack import leastsq, _wrap_jac
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


def fit_rational_to_base_function(rational_func, ref_func, x, degrees=(5, 4), version="A"):
    y = ref_func(x)
    final_params = _curve_fit(rational_func, x, y, degrees=degrees, version=version,
                              maxfev=10000000)[0]
    return np.array(final_params[:degrees[0]+1]), np.array(final_params[degrees[0]+1:])


def find_closest_equivalent(rational_func, new_func, x):
    initials = np.array([1., 0., 1., 0.]) # a, b, c, d
    y = rational_func(x)
    from scipy.optimize import curve_fit
    import torch

    def equivalent_func(x_array, a, b, c, d):
        return a * new_func(c * torch.tensor(x_array) + d) + b
    params = curve_fit(equivalent_func, x, y, initials, bounds=(x.min(), x.max()))
    a, b, c, d = params[0]
    final_func_output = np.array(equivalent_func(x, a, b, c, d))
    final_distance = ((y - final_func_output)**2).sum()
    return (a, b, c, d), final_distance
