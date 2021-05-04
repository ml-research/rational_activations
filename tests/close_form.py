from rational.numpy.rationals import Rational_version_N as RationalNonSafe
from rational.torch import Rational
from numpy.random import normal, seed
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import sys
sns.set_theme()
import inspect
import torch

seed(123)

x_powers = ("", "x", "x²", "x³", "x⁴", "x⁵")


def rational_string(numerator, denominator, rnd_coef=2):
    num_s, deno_s = "", "1"
    numerator = numerator.round(rnd_coef)
    denominator = denominator.round(rnd_coef)
    for coef, x in zip(numerator, x_powers[:len(numerator)]):
        if coef > 0:
            num_s += f"+{coef}{x}"
        elif coef < 0:
            num_s += f"{coef}{x}"
    for coef, x in zip(denominator, x_powers[1:len(denominator)]):
        if coef > 0:
            deno_s += f"+{coef}{x}"
        elif coef < 0:
            deno_s += f"{coef}{x}"
    return f"({num_s[1:]})/({deno_s})"


def fit_rat_gd(function, inp, version="N"):
    inp = torch.tensor(inp)
    loss_fn = torch.nn.MSELoss()
    best_loss = np.inf
    rat = Rational(cuda=False, version=version)
    if "random" in sys.argv:
        rat.numerator = torch.nn.Parameter(torch.normal(torch.zeros(5), torch.ones(5)))
        rat.denominator = torch.nn.Parameter(torch.normal(torch.zeros(4), torch.ones(4)))
    exp = function(inp)
    rat_func = rat
    lr = 0.1
    optimizer = torch.optim.SGD(rat.parameters(), lr=lr)
    for i in range(1001):
        out = rat_func(inp)
        optimizer.zero_grad()
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()
    return rat_func


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
    print(f"\n {funcString}")
    y = f(x)
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(np.array([np.ones_like(x), x, x**2, x**3, x**4, -y*x, -y*x**2, -y*x**3]).T, y)
    # clf.fit(np.array([np.ones_like(x), x, x**2, x**3, x**4, -abs(y*x), -abs(y*x**2), -abs(y*x**3)]).T, y)
    num, deno = clf.coef_[:5], clf.coef_[5:]
    r_str = rational_string(num, deno)
    y_rat = RationalNonSafe(x_r, num, deno)
    if "gd" in sys.argv:
        ax.plot(x_r, f(x_r), '--', color="black", alpha=0.6)
        for version in ["A", "B", "N"]:
            rat = fit_rat_gd(f, x, version)
            torch.set_printoptions(2)
            print(f"{version}: {rat.numerator}, {rat.denominator}")
            ax.plot(x_r, rat(torch.tensor(x_r)).detach().numpy(), label=f"{version}")
    else:
        ax.plot(x_r, y_rat, 'r', label=f"Rat: {r_str}")
        ax.plot(x_r, f(x_r), '--', color="black", label=f"f: {funcString}", alpha=0.6)
    y_min, y_max = ax.get_ylim()
    if y_min < -100 or y_max > 100:
        ax.set_ylim([-100,100])
    ax.set_title(funcString)
    ax.legend()

if "gd" in sys.argv and not "random" in sys.argv:
    plt.annotate("A: Q = 1 + | b_1 * X | + ... + | b_m * X ^m|\n" + \
                 "B: Q = 1 + | b_1 * X  + ... +  b_m * X ^m|\n" + \
                 "N: Q = 1 + b_1 * X  + ... +  b_m * X ^m\n", (-0.4, -1.1))

if "show" in sys.argv:
    plt.show()
else:
    plt.tight_layout()
    title = "rational_close_form"
    if "gd" in sys.argv:
        title += "_gd"
    if "random" in sys.argv:
        title += "_random"
    plt.savefig(f"{title}.png", dpi=300)
