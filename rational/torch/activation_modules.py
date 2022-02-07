from .functions import ActivationModule
import torch.nn.functional as F

class ReLu(ActivationModule):
    def __init__(self, device):
        function = F.relu
        self.function = function
        super().__init__(function, device)

class LReLu(ActivationModule):
    def __init__(self, device):
        function = F.leaky_relu
        super().__init__(function, device)


class Tanh(ActivationModule):
    def __init__(self, device):
        function = F.tanh
        super().__init__(function, device)
    
class Sigmoid(ActivationModule):
    def __init__(self, device):
        function = F.sigmoid
        super().__init__(function, device)

class GLU(ActivationModule):
    def __init__(self, device, dim=-1):
        function = F.glu
        super().__init__(function, device)