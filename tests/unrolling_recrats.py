import torch
from torch.nn import MSELoss
from rational.torch import Rational
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from copy import deepcopy
sns.set_theme()


fig, axes = plt.subplots(5, 5, sharex=True, figsize=(16, 10))

if "sigmoid" in sys.argv:
    inp = torch.arange(-4., 4., 0.1)
    def npfunc(x):
        return 1 / (1 + np.exp(-x))
    funcstr = "sigmoid"
    exp = torch.sigmoid(inp)
elif "sinus" in sys.argv:
    inp = torch.arange(-10., 10., 0.1)
    def npfunc(x):
        return np.sin(x)
    funcstr = "sinus"
    exp = torch.sin(inp)

elif "crazy" in sys.argv:
    inp = torch.arange(-3., 3., 0.1)
    def npfunc(x):
        return 50*x**2/(1 + 4*x)
    funcstr = "50*x**2/(1 + 4*x)"
    exp = 50*inp**2/(1 + 4*inp)
    for ax in axes.flatten():
        ax.set_ylim(-3, 3)

else:
    print("Please provide a function to compare to:")
    print("sigmoid, sinus, ...")

for axs, unroll in zip(axes, (1, 2, 3, 4, 5)):
    print("#" * 15)
    print(f"Unroll: {unroll}")
    rat = Rational(cuda=False)
    numerator = deepcopy(rat.numerator)
    denominator = deepcopy(rat.denominator)
    if unroll == 1:
        rat_func = rat
    elif unroll == 2:
        def rat_func(inp):
            return rat(rat(inp))
    elif unroll == 3:
        def rat_func(inp):
            return rat(rat(rat(inp)))
    elif unroll == 4:
        def rat_func(inp):
            return rat(rat(rat(rat(inp))))
    elif unroll == 5:
        def rat_func(inp):
            return rat(rat(rat(rat(rat(inp)))))
    loss_fn = MSELoss()
    min_loss = np.inf
    min_lr = None
    for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        rat.numerator = deepcopy(numerator)
        rat.denominator = deepcopy(denominator)
        optimizer = torch.optim.SGD(rat.parameters(), lr=lr)
        for i in range(1001):
            out = rat_func(inp)
            optimizer.zero_grad()
            loss = loss_fn(out, exp)
            loss.backward()
            optimizer.step()
        if loss.item() < min_loss:
            min_lr = lr
            min_loss = loss.item()
            print(f'{lr}: {loss.item()}')
    print(f"Taking lr {min_lr} with loss {min_loss}")
    rat.numerator = deepcopy(numerator)
    rat.denominator = deepcopy(denominator)
    optimizer = torch.optim.SGD(rat.parameters(), lr=min_lr)
    for i in range(1001):
        out = rat_func(inp)
        optimizer.zero_grad()
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()
        if not i % 250:
            ax = axs[i // 250]
            ax.set_title(f"epoch {i}")
            if i // 250 == 0:
                ax.plot(inp.detach().numpy(), rat_func(inp).detach().numpy(), label=f"{unroll}-recrat")
                ax.plot(inp.detach().numpy(), npfunc(inp.detach().numpy()), label=funcstr)
                ax.legend()
            else:
                ax.plot(inp.detach().numpy(), rat_func(inp).detach().numpy())
                ax.plot(inp.detach().numpy(), npfunc(inp.detach().numpy()))
            if i // 250 == 4:
                ax.plot([0], [0], alpha=0, label=f"Loss: {round(loss.item(), 4)}")
                ax.plot([0], [0], alpha=0, label=f"lr: {round(min_lr, 4)}")
                ax.legend()
    print(f"Loss: {round(loss.item(), 4)}")

if "show" in sys.argv:
    plt.show()
else:
    if funcstr == "50*x**2/(1 + 4*x)":
        funcstr = "crazy"
    plt.tight_layout()
    plt.savefig(f"{funcstr}.png", dpi=300)

# backward_test(False, "A", True)
# backward_test_hist(True, "A", False)
