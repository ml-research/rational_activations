from mxnet.gluon.nn import HybridSequential, Activation
from rational.torch import Rational as RationalPyTorch
from rational.mxnet import Rational as RationalMxNet
import torch.nn as nn

activations = {nn.ReLU: 'relu', nn.LeakyReLU: 'leaky_relu', nn.Tanh: 'tanh', nn.Sigmoid: 'sigmoid', nn.GELU: 'gelu', nn.Hardswish: 'swish'}

def convert_pytorch_model_to_rational(model, rational_version='A', rational_cuda=False):
    converted = nn.Sequential()
    for name, layer in model.named_children():
        childs = layer.children()
        if len(list(childs)) > 0:
            sequential = nn.Sequential()
            for n, l in layer.named_children():
                sequential.add_module(*_convert_pytorch_layer(n, l, version=rational_version, cuda=rational_cuda))
            converted.add_module(name, sequential)
        else:
            converted.add_module(*_convert_pytorch_layer(name, layer, version=rational_version, cuda=rational_cuda))
    return converted
        

def _convert_pytorch_layer(name, layer, version, cuda):
    for activation in activations:
        if isinstance(layer, activation):
            return f'Rational_{name}', RationalPyTorch(version=version, approx_func=activations[activation], cuda=cuda)
    return name, layer
        
    
def convert_mxnet_model_to_rational(model, rational_version='A', rational_device=None):
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
