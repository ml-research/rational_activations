from rational.torch import Rational
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
from pprint import pprint as print

capturing_epochs = [0, 1, 2, 4, 8, 16, 32, 64, 99]

Rational("leaky_relu", cuda=False)
Rational("sigmoid", cuda=False)
Rational("identity", cuda=False)

print(Rational.list)

if "save" in sys.argv:
    Rational.save_all_graphs(other_func=torch.sin, title="All rationals")
else:
    Rational.show_all(other_func=torch.sin, title="All rationals")

# ================================================


dev = Rational.list[0].device
inp = torch.arange(-3, 3, 0.01).to(dev)

exp = torch.sin(inp)
criterion = torch.nn.MSELoss()

for rat in Rational.list:
    optimizer = torch.optim.SGD(rat.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        optimizer.zero_grad()
        out = rat(inp)
        loss = criterion(out, exp)
        loss.backward()
        if epoch in capturing_epochs:
            rat.capture()
        optimizer.step()

rat.save_animated_graph("coucou.gif")
# import ipdb; ipdb.set_trace()
# Rational.show_all(other_func=torch.sin)
