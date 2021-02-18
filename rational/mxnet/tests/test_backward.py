"""
This file consists test for the backward pass of rational activation functions.
I.e. it is tested here whether they are in fact trainable

This code is largely adapted from the official MxNet MNIST tutorial for Python
(https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html,
accessed on Feb 18, 2021)
"""
from __future__ import print_function

import numpy as np
import mxnet as mx
import mxnet.ndarray as F

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

from rational.mxnet import Rational

# Fixing the random seed
mx.random.seed(42)

# load MNIST set
mnist = mx.test_utils.get_mnist()

batch_size = 10
train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


def _test_backward_template(version: str, function_is_layer: bool):
    """
    This method serves as a base method for all backward tests.
    It tests for a given version of Rational whether Rational can be integrated into a small
    tensorflow model. It also tests whether the rational activation function's coefficients
    (weights) are updated

    :param version: version of the rational activation function
    :param function_is_layer: declare function as a tf.keras.layer in the model if True.
    Else declare as the activation parameter for a given tf.keras.layer
    """
    # create our rational activation function under test
    fut = Rational(version=version)

    # check that the coefficients of the Rational module are in fact trainable
    assert fut.training

    # define network
    net = nn.Sequential()
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))

    gpus = mx.test_utils.list_gpus()
    ctx = [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.02})

    # TODO nums_before_training = fut.numerator.data(mx.current_context())
    # TODO dens_before_training = fut.denominator.data(mx.current_context())

    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # back-propagate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])

        # exit after first loop
        pass

    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc: %s=%f' % (name, acc))

    # copy the new coefficient values
    # TODO nums_after_training = fut.numerator.data(mx.current_context())
    # TODO dens_after_training = fut.denominator.data(mx.current_context())

    # check that at least one coefficient changed in numerators
    # TODO assert not np.all(np.equal(nums_before_training.asnumpy(), nums_after_training.asnumpy()))
    # check that at least one coefficient changed in denominators
    # TODO assert not np.all(np.equal(dens_before_training.asnumpy(), dens_after_training.asnumpy()))


def test_a_backward_as_layer():
    _test_backward_template('A', True)


def test_b_backward_as_layer():
    _test_backward_template('B', True)


def test_c_backward_as_layer():
    _test_backward_template('C', True)


def test_d_backward_as_layer():
    _test_backward_template('D', True)


def test_a_backward_as_param():
    _test_backward_template('A', False)


def test_b_backward_as_param():
    _test_backward_template('B', False)


def test_c_backward_as_param():
    _test_backward_template('C', False)


def test_d_backward_as_param():
    _test_backward_template('D', False)
