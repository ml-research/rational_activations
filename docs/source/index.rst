=================================================
Welcome to Padé Activation Units's documentation!
=================================================

.. highlight:: python

Padé Activation Units are learnable rational activation function to create
rational neural networks.
So far, only the pytorch implementation is available.

Requirements:
#############
This project depends on:

- pytorch
- numpy, scipy (if you want to add different initially approximated functions)

Download and install:
You can download from the
`Github <https://github.com/ml-research/activation_functions>`_ repository or:

::

    pip3 install pau

To use it:

.. literalinclude:: tutorials/code/how_to_use_pau.py
   :lines: 1-6

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:
    :glob:

    tutorials/*


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   :glob:

   *

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
