from mxnet.gluon.nn import HybridSequential, Activation
from rational.mxnet import Rational as RationalMxNet

from rational.torch import Rational as RationalPyTorch
import torch.nn as nn
import copy


activations = {nn.ReLU: 'relu', nn.LeakyReLU: 'leaky_relu', nn.Tanh: 'tanh', nn.Sigmoid: 'sigmoid', nn.GELU: 'gelu', nn.Hardswish: 'swish'}


def convert_pytorch_model_to_rational(model, rational_version='A', rational_cuda=False):
    m = copy.deepcopy(model)
    for n_l, l in m.named_children():
        is_activation = _convert_pytorch_model_to_rational(l, rational_version, rational_cuda)
        if is_activation:
            m._modules[n_l] = _convert_pytorch_layer(l, version=rational_version, cuda=rational_cuda)
    return m


def _convert_pytorch_model_to_rational(m, version, cuda):
    for n_c, c in m.named_children():
        is_activation = _convert_pytorch_model_to_rational(c, version, cuda)
        if is_activation:
            m._modules[n_c] = _convert_pytorch_layer(c, version=version, cuda=cuda)
    return isinstance(m, tuple(activations.keys()))


def _convert_pytorch_layer(layer, version, cuda):
    for activation in activations:
        if isinstance(layer, activation):
            return RationalPyTorch(version=version, approx_func=activations[activation], cuda=cuda)
    raise ValueError("activation function not supported")


def replace_pytorch_activation_functions(model, new_activation_layer):
    m = copy.deepcopy(model)
    for n_l, l in m.named_children():
        is_activation = _replace_pytorch_activation_functions(l, new_activation_layer)
        if is_activation:
            m._modules[n_l] = new_activation_layer()
    return m


def _replace_pytorch_activation_functions(m, new_activation_layer):
    for n_c, c in m.named_children():
        is_activation = _replace_pytorch_activation_functions(c, new_activation_layer)
        if is_activation:
            m._modules[n_c] = new_activation_layer()
    return isinstance(m, tuple(activations.keys()))
        
    
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
