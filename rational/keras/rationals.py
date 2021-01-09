from tensorflow.keras.layers import Layer
import tensorflow as tf

from rational.keras.rational_keras_functions import Rational_KERAS_A_F, Rational_KERAS_B_F, Rational_KERAS_C_F, \
    Rational_KERAS_D_F
from rational.utils.get_weights import get_parameters


class Rational(Layer):
    """
    a class representing rational activation functions for tensorflow, inheriting from tensorflow.keras.layers.Layer
    """

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=False,
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True):
        """
        Inherited from tensorflow.keras.layers.Layer

        Defines custom layer attributes, and creates layer state variables that do not depend on input shapes,
        using ``add_weight()``

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

        self.numerator = self.add_weight(shape=(len(w_numerator),), name='w_numerator',
                                         trainable=trainable and train_numerator,
                                         initializer=tf.keras.initializers.Constant(w_numerator))
        self.denominator = self.add_weight(shape=(len(w_denominator),), name='w_denominator',
                                           trainable=trainable and train_numerator,
                                           initializer=tf.keras.initializers.Constant(w_denominator))

        # set correct rational activation function version
        fun_dict = {'A': Rational_KERAS_A_F, 'B': Rational_KERAS_B_F, 'C': Rational_KERAS_C_F, 'D': Rational_KERAS_D_F}
        self.rational_func = fun_dict.get(version)
        if self.rational_func is None:
            raise ValueError("rational activation function version %s not implemented" % version)

    def build(self, input_shape):
        """
        Inherited from tensorflow.keras.layers.Layer

        This method can be used to create weights that depend on the shape(s) of the input(s),
        using ``add_weight()``. ``__call__()`` will automatically build the layer (if it has not been built yet)
        by calling ``build()``.

        :param input_shape: Instance of `TensorShape`, or list of instances of `TensorShape` if the layer expects a list
         of inputs (one instance per input).
        """
        super(Rational, self).build(input_shape)

    def call(self, inputs, training=True):
        """
        Inherited from tensorflow.keras.layers.Layer

        Called in ``__call__`` after making sure ``build()`` has been called. ``call()`` performs the logic of applying
        the layer to the input tensors (which should be passed in as argument). Two reserved keyword arguments you can
        optionally use in ``call()`` are:

        - training (boolean, whether the call is in inference mode or training mode)
        - mask (boolean tensor encoding masked timesteps in the input, used in RNN layers)

        :param inputs: Input tensor, or list/tuple of input tensors.
        :param training: TODO
        :return: A tensor or list/tuple of tensors.
        """
        return self.rational_func(inputs, self.numerator, self.denominator, training)
