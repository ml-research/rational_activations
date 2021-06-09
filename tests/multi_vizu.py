from rational.torch import Rational
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
from pprint import pprint as print

capturing_epochs = [0, 1, 2, 4, 8, 16, 32, 64, 99]
device = "cuda" if torch.cuda.is_available() else "cpu"

Rational("leaky_relu", cuda="cuda" in device)
Rational("sigmoid", cuda="cuda" in device)
Rational("identity", cuda="cuda" in device)

print(Rational.list)

if "save" in sys.argv:
    Rational.save_all_graphs(other_func=torch.sin, title="All rationals")
else:
    Rational.show_all(other_func=torch.sin, title="All rationals")


# ================================================



criterion = torch.nn.MSELoss()

# for rat in Rational.list:
#     optimizer = torch.optim.SGD(rat.parameters(), lr=0.01, momentum=0.9)
#     for epoch in range(100):
#         inp = (torch.randn(10000)).to(device)
#         exp = torch.sin(inp)
#         optimizer.zero_grad()
#         if epoch in capturing_epochs:
#             rat.saving_input = True
#         out = rat(inp)
#         loss = criterion(out, exp)
#         loss.backward()
#         if epoch in capturing_epochs:
#             # rat.capture(f"Epoch {epoch}")
#             rat.capture(f"Epoch {epoch}")
#             rat.saving_input = False
#         optimizer.step()

optimizers = [torch.optim.SGD(rat.parameters(), lr=0.01, momentum=0.9)
              for rat in Rational.list]
for epoch in range(100):
    for (rat, optimizer) in zip(Rational.list, optimizers):
        inp = ((torch.rand(10000)-0.5)*5).to(device)
        exp = torch.sin(inp)
        optimizer.zero_grad()
        if epoch in capturing_epochs:
            rat.saving_input = True
        out = rat(inp)
        loss = criterion(out, exp)
        loss.backward()
        if epoch in capturing_epochs:
            rat.capture(f"Epoch {epoch}")
            rat.saving_input = False
        optimizer.step()
    # Rational.capture_all()


# seulement la premiere rationelle fait des histogrames.

# rat.save_animated_graph("coucou.gif")
# Rational.save_all_animated_graphs("coucou.gif")
# Rational.export_graphs("coucou.png")
# animated = False
# together = False
for animated in [True, False]:
    for together in [True, False]:
        title = f"rat_a_{animated}_t_{together}"
        Rational.export_evolution_graphs(title, animated=animated, together=together,
                                         other_func=torch.sin)
# import ipdb; ipdb.set_trace()
# Rational.show_all(other_func=torch.sin)
