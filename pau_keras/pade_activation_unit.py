from tensorflow.python.keras.engine.base_layer import Layer

from pau.Constants import *
from pau_keras.pade_keras_functions import *


class PAU(Layer):
    def __init__(self, w_numerator=init_w_numerator, w_denominator=init_w_denominator, center=center, version="A",
                 trainable=True, train_center=True, train_numerator=True, train_denominator=True):
        super(PAU, self).__init__()
        self.center = tf.Variable(initial_value=center, trainable=trainable and train_center)
        self.numerator = tf.Variable(initial_value=w_numerator, trainable=trainable and train_numerator)
        self.denominator = tf.Variable(initial_value=w_denominator, trainable=trainable and train_denominator)

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

        self.pau_func = pau_func

    def build(self, input_shape):
        pass

    def call(self, inputs, training=True):
        return self.pau_func(inputs + self.center, self.numerator, self.denominator, training)
