import torch
from torch.nn import MSELoss
from rational.torch import Rational
import numpy as np

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def backward_test(cuda, version, recurrent_rat):
    inp = torch.arange(-4., 4., 0.1)
    if cuda:
        inp = inp.cuda()
    exp = torch.sigmoid(inp)
    rat = Rational(cuda=cuda)
    if recurrent_rat:
        def rat_func(inp):
            return rat(rat(inp))
    else:
        rat_func = rat
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(rat.parameters(), lr=0.05, momentum=0.9)
    for i in range(200):
        out = rat_func(inp)
        optimizer.zero_grad()
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()
    # rat.show(other_func=sigmoid_np)

for cuda in [True, False]:
    for version in ["A", "B", "C", "D"]:
        for recurrence in [False, True]:
            backward_test(cuda, version, recurrence)
