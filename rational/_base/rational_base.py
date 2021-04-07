import numpy as np


class Rational_base():
    def __init__(self):
        super().__init__()
        self._handle_retrieve_mode = None
        self.distribution = None
        self.best_fitted_function = None
        self.best_fitted_function_params = None

    def show(self, x=None, fitted_function=True, display=True,
             other_func=None, tolerance=0.001, exclude_zero=False):
        """
        Show the function using `matplotlib`.

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                display (bool):
                    If ``True``, displays the graph.
                    Otherwise, returns a dictionary with functions informations. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                tolerance (float):
                    Tolerance the bins frequency.
                    If tolerance is 0.001, every frequency smaller than 0.001. will be cutted out of the histogram.\n
                    Default ``True``
                other_func
        """
        np_func = self.numpy()
        freq = None
        if x is None and self.distribution is None:
            input_range = np.arange(-3, 3, 0.01)
        elif self.distribution is not None and len(self.distribution.bins) > 0:
            freq, bins = _cleared_arrays(self.distribution, tolerance)
            if freq is not None:
                input_range = np.array(bins, dtype=float)
        else:
            input_range = np.array(x, dtype=float)
        outputs = np_func(input_range)
        other_funcs_dict = {}
        if other_func is not None:
            if type(other_func) is dict:
                for func_label, func in other_func.items():
                    other_funcs_dict[func_label] = func(x)
            else:
                if type(other_func) is not list:
                    other_func = [other_func]
                for func in other_func:
                    if '__name__' in dir(func):
                        func_label = func.__name__
                    else:
                        func_label = str(func)
                    other_funcs_dict[func_label] = func(x)
        if display:
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                sns.set_style("whitegrid")
            except ImportError:
                print("Seaborn not found on computer, install it for better",
                      "visualisation")
            ax = plt.gca()
            if freq is not None:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                grey_color = (0.5, 0.5, 0.5, 0.6)
                if exclude_zero:
                    bins = bins[1:]
                    freq = freq[1:]
                ax2.bar(bins, freq, width=bins[1] - bins[0],
                        color=grey_color, edgecolor=grey_color)
            ax.plot(input_range, outputs, label="Rational (self)")
            for func, func_output in other_funcs_dict.items():
                ax.plot(input_range, func_output, label=func)
            if self.best_fitted_function is not None and other_func is None:
                if '__name__' in dir(self.best_fitted_function):
                    func_label = self.best_fitted_function.__name__
                else:
                    func_label = str(self.best_fitted_function)
                a, b, c, d = self.best_fitted_function_params
                result = a * self.best_fitted_function(c * input_range + d) + b
                ax.plot(input_range, result, "r-", label=f"Fitted {func_label}")
            ax.legend()
            plt.show()
        else:
            if freq is None:
                hist_dict = None
            else:
                hist_dict = {"bins": bins, "freq": freq,
                             "width": bins[1] - bins[0]}
            if "best_fitted_function" not in dir(self) or self.best_fitted_function is None:
                fitted_function = None
            else:
                a, b, c, d = self.best_fitted_function_params
                result = a * self.best_fitted_function(c * input_range + d) + b
                fitted_function = {"function": self.best_fitted_function,
                                   "params": (a, b, c, d),
                                   "y": result}
            return {"hist": hist_dict,
                    "line": {"x": input_range, "y": outputs},
                    "fitted_function": fitted_function,
                    "other_func": other_funcs_dict}

    def fit(self, function, x=None, show=False):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent \
        function of the provided one as close as possible to this rational \
        function.

        Arguments:
                function (callable):
                    The function you want to fit to rational.\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together.\n
                    Default ``True``
                show (bool):
                    If  ``True``, plots the final fitted function and \
                    rational (using matplotlib).\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function \
                (vertical and horizontal scales and bias) \n
            dist: The final distance between the rational function and the \
            fitted one
        """
        if "rational.keras" in str(type(function)) or \
           "rational.torch" in str(type(function)):
            function = function.numpy()
        used_dist = False
        rational_numpy = self.numpy()
        if x is not None:
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        else:
            if self.distribution is not None:
                freq, bins = _cleared_arrays(self.distribution)
                x = bins
                used_dist = True
            else:
                import numpy as np
                x = np.arange(-3., 3., 0.1)
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        if show:
            def func(inp):
                return a * function(c * inp + d) + b

            if '__name__' in dir(function):
                func_label = function.__name__
            else:
                func_label = str(function)
            self.show(x, other_func={func_label: func})
        if self.best_fitted_function is None:
            self.best_fitted_function = function
            self.best_fitted_function_params = (a, b, c, d)
        return (a, b, c, d), distance

    def best_fit(self, functions_list, x=None, show=False):
        if self.distribution is not None:
            freq, bins = _cleared_arrays(self.distribution)
            x = bins
        (a, b, c, d), distance = self.fit(functions_list[0], x=x, show=show)
        min_dist = distance
        print(f"{functions_list[0]}: {distance:>3}")
        params = (a, b, c, d)
        final_function = functions_list[0]
        for func in functions_list[1:]:
            (a, b, c, d), distance = self.fit(func, x=x, show=show)
            print(f"{func}: {distance:>3}")
            if min_dist > distance:
                min_dist = distance
                params = (a, b, c, d)
                final_func = func
                print(f"{func} is the new best fitted function")
        self.best_fitted_function = final_func
        self.best_fitted_function_params = params
        return final_func, (a, b, c, d)

    def numpy(self):
        raise NotImplementedError("the numpy method is not implemented for",
                                  " this class, only for the mother class")


def _cleared_arrays(hist, tolerance=0.001):
    freq, bins = hist.normalize()
    first = (freq > tolerance).argmax()
    last = - (freq > tolerance)[::-1].argmax()
    if last == 0:
        return freq[first:], bins[first:]
    return freq[first:last], bins[first:last]
