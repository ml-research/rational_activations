from tensorflow.python.keras.engine.base_layer import Layer

from pau_keras.pade_keras_functions import *

init_w_numerator = [0.09163206842254161,
                    0.5049965361843953,
                    0.7451269450654521,
                    0.45290252844805917,
                    0.11193217514133497,
                    0.010166300350864386]

init_w_denominator = [-9.411908117465748e-07,
                      0.896831954111865,
                      2.070929650028975e-06,
                      0.020131018048667716]


class PAU(Layer):
    def __init__(self, w_numerator=init_w_numerator, w_denominator=init_w_denominator, version="B", trainable=True):
        super(PAU, self).__init__()
        self.numerator = tf.Variable(initial_value=w_numerator, trainable=trainable)
        self.denominator = tf.Variable(initial_value=w_denominator, trainable=trainable)

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
        return self.pau_func(inputs, self.numerator, self.denominator, training)
