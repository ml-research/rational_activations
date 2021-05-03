from rational.torch import Rational
import matplotlib.pyplot as plt
import seaborn as sns
import torch

Rational("leaky_relu", cuda=False)
Rational("sigmoid", cuda=False)
Rational("identity", cuda=False)

print(Rational.list)
with sns.axes_style("whitegrid"):
    fig, axes = plt.subplots(3, 1)

Rational.show_all(other_func=torch.sin, axes=axes)
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
        optimizer.step()

with sns.axes_style("whitegrid"):
    fig, naxes = plt.subplots(3, 1)
Rational.show_all(other_func=torch.sin, axes=naxes)
