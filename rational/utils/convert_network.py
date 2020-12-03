from rational.torch import Rational as RationalPyTorch
import torch.nn as nn

# mapping of known activation functions
activations = {nn.ReLU: 'relu', nn.LeakyReLU: 'leaky_relu', nn.Tanh: 'tanh', nn.Sigmoid: 'sigmoid', nn.GELU: 'gelu',
               nn.Hardswish: 'swish'}


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
    # handle default here
    return name, layer
