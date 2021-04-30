from rational.numpy.rationals import Rational_version_N as RationalNonSafe
from numpy.random import normal, seed
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import sys
sns.set_theme()
import inspect

seed(123)


N = 1000
x = normal(size=N)
x_r = np.arange(-3, 3, 0.05)

rat = RationalNonSafe
f1 = lambda x : 50*x**2/(1 + 4*x)
f2 = lambda x : (100 - 50*x - 100*x**2)/(1 - 10*x - 10*x**2)
f3 = lambda x : (100 - 50*x - 100*x**2)/(1 - 10*x - 5*x**2)
f4 = lambda x : np.sin(x)

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(16, 8))
for f, ax in zip([f1, f2, f3, f4], axs.flatten()):
    funcString = "f(x) =" + str(inspect.getsourcelines(f)[0][0]).split(":")[1]
    y = f(x)
    clf = linear_model.LinearRegression(fit_intercept=False)
    # clf.fit(np.array([np.ones_like(x), x, x**2, x**3, x**4, -np.abs(y*x), -np.abs(y*x**2), -np.abs(y*x**3)]).T, y)
    clf.fit(np.array([np.ones_like(x), x, x**2, x**3, x**4, -y*x, -y*x**2, -y*x**3]).T, y)
    num, deno = clf.coef_[:5], clf.coef_[5:]
    print(f"{num =}")
    print(f"{deno =}")
    y_rat = RationalNonSafe(x_r, num, deno)
    ax.plot(x_r, y_rat, 'r', label="Rat")
    ax.plot(x_r, f(x_r), '--', color="orange", label="f", alpha=0.4)
    ax.plot(x, y, 'o', color="black", markersize=2, label="samples")
    y_min, y_max = ax.get_ylim()
    if y_min < -100 or y_max > 100:
        ax.set_ylim([-100,100])
    ax.set_title(funcString)
    ax.legend()
plt.show()
