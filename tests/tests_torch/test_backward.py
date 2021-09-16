import torch
from rational.torch import Rational

def _template_backward(version, cuda):
    rat = Rational(version=version, cuda=cuda)
    if cuda:
        device = "cuda"
    else:
        device = "cpu"
    inp = torch.arange(-2, 2, 0.1).to(device)
    expected_output = torch.sigmoid(inp)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 0.1
    optimizer = torch.optim.Adam(rat.parameters(), lr=learning_rate)
    loss_save = 10000.
    for i in range(1000):
        output = rat(inp)
        loss = loss_fn(expected_output, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            assert loss < loss_save  # assert convergeance
            loss_save = loss


def test_backward_A_cpu():
    _template_backward("A", False)


def test_backward_B_cpu():
    _template_backward("B", False)


def test_backward_C_cpu():
    _template_backward("C", False)


def test_backward_A_gpu():
    _template_backward("A", True)


def test_backward_B_gpu():
    _template_backward("B", True)


def test_backward_C_gpu():
    _template_backward("C", True)
