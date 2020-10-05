import torch.nn as nn
from torch.cuda import is_available as torch_cuda_available
from pau.Constants import *
from pau.utils import get_parameters


if torch_cuda_available():
    try:
        from pau_torch.pade_cuda_functions import *
    except:
        print('error importing pade_cuda, is cuda not avialable?')

from pau_torch.pade_pytorch_functions import *


class PAU(nn.Module):

    def __init__(self, approx_func="leaky_relu", degrees=(5,4), cuda=None,
                 version="A", trainable=True, train_center=True, train_numerator=True, train_denominator=True):
        super(PAU, self).__init__()

        if cuda is None:
            cuda = torch_cuda_available()
        device = "cuda" if cuda else "cpu"

        center, w_numerator, w_denominator = get_parameters(version, degrees, approx_func)

        self.center = nn.Parameter(torch.FloatTensor([center]).to(device), requires_grad=trainable and train_center)
        self.numerator = nn.Parameter(torch.FloatTensor(w_numerator).to(device), requires_grad=trainable and train_numerator)
        self.denominator = nn.Parameter(torch.FloatTensor(w_denominator).to(device), requires_grad=trainable and train_denominator)
        self.degrees = degrees
        self.version = version
        self.training = trainable

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

    def __repr__(self):
        desc = f"Pade Activation Unit (version {self.version}) of degrees {self.degrees} running on {self.center.device}"
        return desc

    def cpu(self):
        if self.version == "A":
            pau_func = PAU_PYTORCH_A_F
        elif self.version == "B":
            pau_func = PAU_PYTORCH_B_F
        elif self.version == "C":
            pau_func = PAU_PYTORCH_C_F
        elif self.version == "D":
            pau_func = PAU_PYTORCH_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)
        self.activation_function = pau_func
        return super().cpu()

    def cuda(self):
        if self.version == "A":
            pau_func = PAU_CUDA_A_F
        elif self.version == "B":
            pau_func = PAU_CUDA_B_F
        elif self.version == "C":
            pau_func = PAU_CUDA_C_F
        elif self.version == "D":
            pau_func = PAU_CUDA_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)

        self.activation_function = pau_func.apply
        return super().cuda()
