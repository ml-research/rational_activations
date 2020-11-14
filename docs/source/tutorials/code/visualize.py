from rational_torch import Rational
rational_function = Rational("tanh") # Initialized closed to tanh

rational_function.show()


import numpy as np
rational_function.show(np.arange(-6,12, 2))

rational_function.input_retrieve_mode()
# Retrieving input from now on.

import torch
means = torch.ones((50, 50)) * 2.
stds = torch.ones((50, 50)) * 3.
for _ in range(1500):
    input = torch.normal(means, stds).to(rational_function.device)
    rational_function(input)

# Training mode, no longer retrieving the input.

rational_function.show()
