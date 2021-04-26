import torch
from torch.nn import MSELoss
from rational.torch import Rational
import numpy as np

vizu_epochs = [0, 2, 4, 7, 10, 50, 100, 200]

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
        if not i :
            rat.show(other_func=sigmoid_np)


def backward_test_hist(cuda, version, recurrent_rat):
    inp = torch.arange(-4.05, 4., 0.1)
    if cuda:
        inp = inp.cuda()
    exp = torch.sigmoid(inp)
    rat = Rational(cuda=cuda)
    rat.input_retrieve_mode()
    if recurrent_rat:
        def rat_func(inp):
            return rat(rat(inp))
    else:
        rat_func = rat
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(rat.parameters(), lr=0.05, momentum=0.9)
    for i in range(201):
        if i in vizu_epochs:
            rat.clear_hist()
        out = rat_func(inp)
        optimizer.zero_grad()
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()
        if i in vizu_epochs:
            rat.snapshot(f"Epoch {i}", other_func=sigmoid_np)
    for snap in rat.snapshot_list:
        snap.show(other_func=sigmoid_np)
    import ipdb; ipdb.set_trace()

# for cuda in [True, False]:
#     for version in ["A", "B", "C", "D"]:
#         for recurrence in [False, True]:
#             backward_test(cuda, version, recurrence)

# backward_test_hist(False, "A", False)
backward_test_hist(True, "A", False)
