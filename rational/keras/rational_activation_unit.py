from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf

from rational.keras.rational_keras_functions import Rational_KERAS_A_F, Rational_KERAS_B_F, Rational_KERAS_C_F, \
    Rational_KERAS_D_F
from rational.utils.get_weights import get_parameters
from rational.keras import *


class Rational(Layer):
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=False,
                 version="A", trainable=True, train_numerator=True, train_denominator=True):
        super(Rational, self).__init__()

        w_numerator, w_denominator = get_parameters(
            version, degrees, approx_func)
        self.numerator = tf.Variable(
            initial_value=w_numerator, trainable=trainable and train_numerator)
        self.denominator = tf.Variable(
            initial_value=w_denominator, trainable=trainable and train_denominator)

        if version == "A":
            rational_func = Rational_PYTORCH_A_F
        elif version == "B":
            rational_func = Rational_PYTORCH_B_F
        elif version == "C":
            rational_func = Rational_PYTORCH_C_F
        elif version == "D":
            rational_func = Rational_PYTORCH_D_F
        else:
            raise ValueError("version %s not implemented" % version)

        self.rational_func = rational_func

    def build(self, input_shape):
        pass

    def call(self, inputs, training=True):
        return self.rational_func(inputs, self.numerator, self.denominator, training)
