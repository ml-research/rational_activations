from mxnet.gluon.nn import HybridSequential, Activation
from rational.mxnet import Rational as RationalMxNet

from rational.torch import Rational as RationalPyTorch
import torch.nn as nn
import copy


activations = {nn.ReLU: 'relu', nn.LeakyReLU: 'leaky_relu', nn.Tanh: 'tanh', nn.Sigmoid: 'sigmoid', nn.GELU: 'gelu', nn.Hardswish: 'swish'}


def convert_pytorch_model_to_rational(model, rational_version='A', rational_cuda=True):
    m = copy.deepcopy(model)
    _recursive_pytorch_conversion(m, rational_version, rational_cuda)
    return m


def _recursive_pytorch_conversion(module, rational_version, rational_cuda):
    for attr_str in dir(module):
        _convert_pytorch_layer(module, attr_str, rational_version, rational_cuda)
    
    for child in module.children():
        _recursive_pytorch_conversion(child, rational_version, rational_cuda)
        

def _convert_pytorch_layer(module, attr_str, version, cuda):
    at = getattr(module, attr_str)
    for activation in activations:
        if isinstance(at, activation):
            act = RationalPyTorch(version=version, approx_func=activations[activation], cuda=cuda)
            setattr(module, attr_str, act)
            break
        
    
def convert_mxnet_model_to_rational(model, rational_version='A', rational_device=None):
    model = copy.deepcopy(model)
    converted = HybridSequential()
    for name, layer in model._children.items():
        childs = layer._children.items()
        if len(list(childs)) > 0:
            seq = HybridSequential()
            for n, l in layer._children.items():
                seq.add(_convert_mxnet_layer(layer, rational_version, rational_device))
            converted.add(seq)
        else:
            converted.add(_convert_mxnet_layer(layer, rational_version, rational_device))
    return converted


def _convert_mxnet_layer(layer, version, device):
    if isinstance(layer, Activation):
        return RationalMxNet(version=version, approx_func='relu', device=device)
    return layer
