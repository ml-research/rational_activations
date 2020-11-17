Make a rational Network in Pytorch
==================================

To use Rational in mxnet, you can import the Rational module and instantiate a
rational function:

.. literalinclude:: code/how_to_use_rationals_mxnet.py
   :lines: 1-7

depending on CUDA available on the machine.

To place the rational function on the cpu/gpu:

.. literalinclude:: code/how_to_use_rationals_mxnet.py
   :lines: 9-10

To inspect the parameter of the rational function

.. literalinclude:: code/how_to_use_rationals_mxnet.py
    :lines: 12-17

If you now want to create a mxnet Rational Network class:

.. literalinclude:: code/how_to_use_rationals_mxnet.py
    :lines: 19-61

Now we can instantiate a Rational Network and a Recurrent Rational Network and
pass them inputs.

.. literalinclude:: code/how_to_use_rationals_mxnet.py
    :lines: 63-85

To see the activation function:

.. literalinclude:: code/how_to_use_rationals_mxnet.py
    :lines: 87-98

.. image:: ../approx_lrelu.png
   :width: 600
