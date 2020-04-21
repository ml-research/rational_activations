import torch.nn as nn

from pau.Constants import *

try:
    from pau_torch.pade_cuda_functions import *
except:
    print('error importing pade_cuda, is cuda not avialable?')

from pau_torch.pade_pytorch_functions import *


class PAU(nn.Module):

    def __init__(self, w_numerator=init_w_numerator, w_denominator=init_w_denominator, center=center, cuda=True,
                 version="A", trainable=True, train_center=True, train_numerator=True, train_denominator=True):
        super(PAU, self).__init__()

        self.center = nn.Parameter(torch.FloatTensor([center]), requires_grad=trainable and train_center)
        self.numerator = nn.Parameter(torch.FloatTensor(w_numerator), requires_grad=trainable and train_numerator)
        self.denominator = nn.Parameter(torch.FloatTensor(w_denominator), requires_grad=trainable and train_denominator)

        if cuda:
            if version == "A":
                pau_func = PAU_CUDA_A_F
            elif version == "B":
                pau_func = PAU_CUDA_B_F
            elif version == "C":
                pau_func = PAU_CUDA_C_F
            elif version == "D":
                pau_func = PAU_CUDA_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = pau_func.apply
        else:
            if version == "A":
                pau_func = PAU_PYTORCH_A_F
            elif version == "B":
                pau_func = PAU_PYTORCH_B_F
            elif version == "C":
                pau_func = PAU_PYTORCH_C_F
            elif version == "D":
                pau_func = PAU_PYTORCH_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = pau_func

    def forward(self, x):
        out = self.activation_function(x + self.center, self.numerator, self.denominator, self.training)
        return out
