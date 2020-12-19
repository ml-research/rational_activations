from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf


from rational.utils.get_weights import get_parameters


class Rational(Layer):
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=False,
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True):
        """
        Rational activation function inherited from tensorflow keras ``Layer``

        Arguments:
                approx_func (str):
                    The name of the approximated function for initialisation. \
                    The different initialable functions are available in \
                    `rational.rationals_config.json`. \n
                    Default ``leaky_relu``.
                degrees (tuple of int):
                    The degrees of the numerator (P) and denominator (Q).\n
                    Default ``(5, 4)``
                cuda (bool):
                    Use GPU CUDA version. \n
                    If ``None``, use cuda if available on the machine\n
                    Default ``None``
                version (str):
                    Version of Rational to use. Rational(x) = P(x)/Q(x)\n
                    `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                    `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                    `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                    `D`: like `B` with noise\n
                    Default ``A``
                trainable (bool):
                    If the weights are trainable, i.e, if they are updated during \
                    backward pass\n
                    Default ``True``
        Returns:
            Module: Rational module
        """
        super(Rational, self).__init__()

        w_numerator, w_denominator = get_parameters(version, degrees, approx_func)
        self.numerator = tf.Variable(initial_value=w_numerator, trainable=trainable and train_numerator)
        self.denominator = tf.Variable(initial_value=w_denominator, trainable=trainable and train_denominator)

        if version == "A":
            rational_func = Rational_KERAS_A_F
        elif version == "B":
            rational_func = Rational_KERAS_B_F
        elif version == "C":
            rational_func = Rational_KERAS_C_F
        elif version == "D":
            rational_func = Rational_KERAS_D_F
        else:
            raise ValueError("version %s not implemented" % version)

        self.rational_func = rational_func

    def build(self, input_shape):
        pass

    def call(self, inputs, training=True):
        return self.rational_func(inputs, self.numerator, self.denominator, training)
